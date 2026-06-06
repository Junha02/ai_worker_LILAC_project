"""
training.py

Plain PyTorch training utilities for LILAC.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import HIDDEN_DIM, LATENT_DIM
from data import build_training_arrays, iter_episode_npzs, normalize_episode_type, save_training_arrays
from language import (
    CanonicalLanguageDataset,
    CanonicalLanguageIndex,
    DatasetAlphaLabeler,
    TransformerLanguageEmbedder,
)
from lilac_model import LILACModel


def make_staging_run_dir(run_dir):
    run_dir = Path(run_dir)
    return run_dir.parent / (".%s_staging_%s" % (run_dir.name, uuid.uuid4().hex[:8]))


def prune_lilac_run_siblings(run_dir, prefix="lilac_sh5_right"):
    """
    Keep one latest run directory and remove older LILAC model runs.
    """
    run_dir = Path(run_dir)
    runs_root = run_dir.parent
    if not runs_root.exists():
        return []

    keep = run_dir.resolve()
    removed = []
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith(prefix):
            continue
        if path.resolve() == keep:
            continue
        shutil.rmtree(path)
        removed.append(path)
    return removed


def resolve_all_dataset_dirs(data_root, episode_types=("instruction", "correction")):
    """
    Return every trajectory folder under data/<task>/<episode_type>.
    """
    data_root = Path(data_root)
    wanted_types = (
        None
        if episode_types is None
        else tuple(normalize_episode_type(kind) for kind in episode_types)
    )
    episode_dirs = {}
    for npz_path in iter_episode_npzs(
            data_dir       = data_root,
            tasks          = None,
            episode_types  = wanted_types,
        ):
        task = npz_path.parents[1].name
        episode_type = normalize_episode_type(npz_path.parent.name)
        episode_dirs.setdefault((task, episode_type), set()).add(npz_path.parent.resolve())

    ordered = []
    for key in sorted(episode_dirs):
        ordered.extend(sorted(episode_dirs[key]))

    if not ordered:
        raise FileNotFoundError(
            "No .npz trajectory episodes found under %s. Expected data/<task>/instruction or correction."
            % data_root
        )
    return ordered


def fit_latent_alignment(
        model,
        states,
        language,
        alphas,
        actions,
        joystick_z,
        device,
        batch_size = 512,
        min_joystick_norm = 0.05,
    ):
    """
    Fit a zero-preserving linear map from physical joystick z to model latent z.

    The paper/code learn latent actions from demonstrations and then feed the
    joystick directly to the decoder. That leaves sign/swap/rotation ambiguity;
    this calibration resolves it without changing the LILAC reconstruction loss.
    """
    joystick_z = np.asarray(joystick_z, dtype=np.float32)
    valid = (
        np.isfinite(joystick_z).all(axis=1)
        & (np.linalg.norm(joystick_z, axis=1) >= float(min_joystick_norm))
    )
    if int(valid.sum()) < 4:
        return None

    model.eval()
    z_model = []
    valid_idxs = np.flatnonzero(valid)
    with torch.no_grad():
        for start in range(0, len(valid_idxs), int(batch_size)):
            idxs = valid_idxs[start:start + int(batch_size)]
            state_b = torch.from_numpy(states[idxs]).float().to(device)
            lang_b = torch.from_numpy(language[idxs]).float().to(device)
            alpha_b = torch.from_numpy(alphas[idxs]).float().to(device)
            action_b = torch.from_numpy(actions[idxs]).float().to(device)
            z_b = model.encode_action_latent(state_b, lang_b, alpha_b, action_b)
            z_model.append(z_b.detach().cpu().numpy())

    z_model = np.concatenate(z_model, axis=0).astype(np.float32)
    z_joy = joystick_z[valid_idxs].astype(np.float32)
    weight, _, _, _ = np.linalg.lstsq(z_joy, z_model, rcond=None)
    pred = z_joy @ weight
    residual_mse = float(np.mean((pred - z_model) ** 2))
    return {
        "weight": weight.astype(np.float32),
        "bias": np.zeros((weight.shape[1],), dtype=np.float32),
        "n_valid": int(valid.sum()),
        "residual_mse": residual_mse,
    }


def save_latent_alignment(alignment, path):
    if alignment is None:
        return None
    path = Path(path)
    np.savez_compressed(
        path,
        weight       = alignment["weight"],
        bias         = alignment["bias"],
        n_valid      = np.asarray([alignment["n_valid"]], dtype=np.int64),
        residual_mse = np.asarray([alignment["residual_mse"]], dtype=np.float32),
    )
    return path


def prepare_training_arrays(
        data_dir,
        out_path,
        tasks                 = None,
        episode_types         = None,
        object_state          = None,
        language_dataset_path = None,
        normalize_actions     = True,
    ):
    episode_npzs = list(
        iter_episode_npzs(
            data_dir       = data_dir,
            tasks          = tasks,
            episode_types  = episode_types,
        )
    )
    if language_dataset_path is None:
        language_dataset = CanonicalLanguageDataset.load()
    else:
        language_dataset = CanonicalLanguageDataset.load(language_dataset_path)

    arrays = build_training_arrays(
        episode_npzs       = episode_npzs,
        object_state       = object_state,
        language_dataset   = language_dataset,
        normalize_actions  = normalize_actions,
        alpha_labeler      = DatasetAlphaLabeler(language_dataset),
    )
    return save_training_arrays(arrays, out_path)


def train_lilac_from_arrays(
        arrays_path,
        run_dir,
        language_index_path   = None,
        embedder              = None,
        language_dataset_path = None,
        latent_dim            = LATENT_DIM,
        hidden_dim            = HIDDEN_DIM,
        batch_size            = 512,
        n_epochs              = 50,
        lr                    = 1e-3,
        weight_decay          = 1e-2,
        val_fraction          = 0.1,
        seed                  = 21,
        device                = None,
        verbose               = True,
        overwrite_run         = True,
        prune_existing_runs   = True,
        run_dir_prefix        = "lilac_sh5_right",
    ):
    """
    Train LILAC from arrays saved by prepare_training_arrays.
    """
    rng = np.random.default_rng(seed)
    run_dir = Path(run_dir)
    staging_dir = make_staging_run_dir(run_dir)

    payload = np.load(arrays_path, allow_pickle=True)
    states = np.asarray(payload["states"], dtype=np.float32)
    actions = np.asarray(payload["actions"], dtype=np.float32)
    utterances = [str(u) for u in payload["utterances"].tolist()]
    alphas = np.asarray(payload["alphas"], dtype=np.float32)
    joystick_z = (
        np.asarray(payload["latent_z"], dtype=np.float32)
        if "latent_z" in payload
        else None
    )

    if language_dataset_path is None:
        language_dataset = CanonicalLanguageDataset.load()
    else:
        language_dataset = CanonicalLanguageDataset.load(language_dataset_path)

    language_index = None
    if language_index_path is not None:
        loaded_language_index = CanonicalLanguageIndex.load(language_index_path)
        if (
                loaded_language_index.ids == language_dataset.ids()
                and loaded_language_index.utterances == language_dataset.texts()
            ):
            language_index = loaded_language_index
        elif verbose:
            print("Language index is stale; rebuilding:", language_index_path)

    if language_index is None:
        embedder = embedder if embedder is not None else TransformerLanguageEmbedder(device="cpu")
        language_index = CanonicalLanguageIndex.build(language_dataset, embedder)
    staging_dir.mkdir(parents=True, exist_ok=False)
    staging_language_index_path = staging_dir / "language_index.npz"
    final_language_index_path = run_dir / "language_index.npz"
    try:
        language_index.save(staging_language_index_path)
        lang_cache = {u: language_index.lookup(u)["embedding"] for u in sorted(set(utterances))}
        language = np.asarray([lang_cache[u] for u in utterances], dtype=np.float32)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    n = len(states)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * float(val_fraction)))) if n > 1 else 0
    val_idxs = order[:n_val]
    train_idxs = order[n_val:] if n_val > 0 else order

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LILACModel(
        state_dim    = states.shape[1],
        language_dim = language.shape[1],
        action_dim   = actions.shape[1],
        latent_dim   = latent_dim,
        hidden_dim   = hidden_dim,
    ).to(device)

    def make_loader(idxs, shuffle):
        ds = TensorDataset(
            torch.from_numpy(states[idxs]).float(),
            torch.from_numpy(language[idxs]).float(),
            torch.from_numpy(alphas[idxs]).float(),
            torch.from_numpy(actions[idxs]).float(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idxs, shuffle=True)
    val_loader = make_loader(val_idxs, shuffle=False) if n_val > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    best_val = None
    best_state = None

    removed_run_dirs = []
    try:
        for epoch in range(int(n_epochs)):
            model.train()
            train_losses = []
            for state_b, lang_b, alpha_b, action_b in train_loader:
                state_b = state_b.to(device)
                lang_b = lang_b.to(device)
                alpha_b = alpha_b.to(device)
                action_b = action_b.to(device)
                loss = model.reconstruction_loss(state_b, lang_b, alpha_b, action_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            val_loss = None
            if val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for state_b, lang_b, alpha_b, action_b in val_loader:
                        loss = model.reconstruction_loss(
                            state_b.to(device),
                            lang_b.to(device),
                            alpha_b.to(device),
                            action_b.to(device),
                        )
                        val_losses.append(float(loss.detach().cpu()))
                val_loss = float(np.mean(val_losses))
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
            }
            history.append(row)
            if verbose:
                print("[epoch %03d] train_loss=%.6f val_loss=%s" % (
                    epoch,
                    row["train_loss"],
                    "None" if val_loss is None else "%.6f" % val_loss,
                ))

        if best_state is not None:
            model.load_state_dict(best_state)

        latent_alignment_path = None
        if joystick_z is not None:
            alignment = fit_latent_alignment(
                model      = model,
                states     = states,
                language   = language,
                alphas     = alphas,
                actions    = actions,
                joystick_z = joystick_z,
                device     = device,
                batch_size = batch_size,
            )
            staging_latent_alignment_path = save_latent_alignment(alignment, staging_dir / "latent_alignment.npz")
            if staging_latent_alignment_path is not None:
                latent_alignment_path = run_dir / "latent_alignment.npz"
            if verbose:
                if latent_alignment_path is None:
                    print("Latent alignment skipped: not enough recorded nonzero joystick z samples.")
                else:
                    print("Saved latent alignment:", latent_alignment_path)

        extra_config = {
            "arrays_path": str(arrays_path),
            "canonical_language_index_path": str(final_language_index_path),
            "n_train_samples": int(len(train_idxs)),
            "n_val_samples": int(len(val_idxs)),
            "seed": int(seed),
        }
        if latent_alignment_path is not None:
            extra_config["latent_alignment_path"] = str(latent_alignment_path)

        model.save_bundle(
            staging_dir,
            language_index_path = final_language_index_path,
            extra_config        = extra_config,
        )
        with (staging_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        if prune_existing_runs:
            removed_run_dirs.extend(prune_lilac_run_siblings(run_dir, prefix=run_dir_prefix))
        if run_dir.exists():
            if not overwrite_run:
                raise FileExistsError("Run directory already exists: %s" % run_dir)
            shutil.rmtree(run_dir)
            removed_run_dirs.append(run_dir)
        staging_dir.rename(run_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return {
        "model": model,
        "run_dir": run_dir,
        "history": history,
        "language_index": language_index,
        "removed_run_dirs": removed_run_dirs,
    }
