"""
lilac_model.py

Paper-faithful LILAC model components implemented with plain PyTorch.
"""

from __future__ import annotations

from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import ACTION_DIM, HIDDEN_DIM, LANGUAGE_DIM, LATENT_DIM


MODEL_ARCH_VERSION = "state_conditioned_compressor_v2"


class MLP(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            out_dim,
            activation = None,
        ):
        super().__init__()
        if activation is None:
            activation = nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LILACModel(nn.Module):
    """
    Language-Informed Latent Actions with Corrections.

    Implements the model in paper Section 4.2:

        h_state    = EncodeState(s)
        h_language = EncodeLanguage(u, c)
        alpha      = GPTGating(u, c)
        h_gated    = alpha * h_state + (1 - alpha) * bias
        h_fused    = FiLM(h_gated, h_language)
        B_bases    = GramSchmidt(Projection(h_fused))
        a_hat      = B_bases @ z

    During training, Compress(action) supplies z and the loss is MSE between
    action and reconstructed action.
    """

    def __init__(
            self,
            state_dim,
            language_dim = LANGUAGE_DIM,
            action_dim   = ACTION_DIM,
            latent_dim   = LATENT_DIM,
            hidden_dim   = HIDDEN_DIM,
        ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.language_dim = int(language_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        self.state_bn = nn.BatchNorm1d(self.state_dim)
        self.state_encoder = MLP(self.state_dim, self.hidden_dim, self.hidden_dim)
        self.language_encoder = MLP(self.language_dim, self.hidden_dim, self.hidden_dim)
        self.h_no_context = nn.Parameter(torch.randn(1, self.hidden_dim))

        self.film_gamma = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.film_beta = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)

        self.basis_projection = MLP(
            self.hidden_dim,
            self.hidden_dim,
            self.action_dim * self.latent_dim,
        )
        self.compressor = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh(),
        )

    @staticmethod
    def gram_schmidt_columns(bases, eps=1e-8):
        """
        Orthonormalize basis columns with modified Gram-Schmidt.

        Args:
            bases: Tensor shaped [batch, action_dim, latent_dim].
        """
        q_cols = []
        for i in range(bases.shape[-1]):
            v = bases[:, :, i]
            for q in q_cols:
                coeff = torch.sum(v * q, dim=1, keepdim=True)
                v = v - coeff * q
            v = v / torch.clamp(torch.linalg.norm(v, dim=1, keepdim=True), min=eps)
            q_cols.append(v)
        return torch.stack(q_cols, dim=-1)

    def encode_state(self, state):
        state_norm = self.state_bn(state)
        h_state = self.state_encoder(state_norm)
        return F.normalize(h_state, dim=1)

    def encode_language(self, language_embedding):
        h_language = self.language_encoder(language_embedding)
        return F.normalize(h_language, dim=1)

    def fuse(self, state, language_embedding, alpha):
        h_state = self.encode_state(state)
        h_language = self.encode_language(language_embedding)
        h_no_context = F.normalize(self.h_no_context, dim=1).expand_as(h_state)

        alpha = alpha.reshape(-1, 1).to(dtype=h_state.dtype, device=h_state.device)
        h_gated = alpha * h_state + (1.0 - alpha) * h_no_context

        gamma = self.film_gamma(h_language)
        beta = self.film_beta(h_language)
        h_fused = gamma * h_gated + beta
        return h_fused

    def bases(self, state, language_embedding, alpha):
        h_fused = self.fuse(state, language_embedding, alpha)
        bases = self.basis_projection(h_fused)
        bases = bases.reshape(-1, self.action_dim, self.latent_dim)
        return self.gram_schmidt_columns(bases)

    def decoder(self, state, language_embedding, alpha, z):
        bases = self.bases(state, language_embedding, alpha)
        z = z.reshape(-1, self.latent_dim, 1).to(dtype=bases.dtype, device=bases.device)
        return torch.bmm(bases, z).squeeze(-1)

    def encode_action_latent(self, state, language_embedding, alpha, action):
        h_fused = self.fuse(state, language_embedding, alpha)
        action = action.reshape(-1, self.action_dim).to(dtype=h_fused.dtype, device=h_fused.device)
        return self.compressor(torch.cat([h_fused, action], dim=1))

    def forward(self, state, language_embedding, alpha, action):
        z = self.encode_action_latent(state, language_embedding, alpha, action)
        return self.decoder(state, language_embedding, alpha, z)

    def reconstruction_loss(self, state, language_embedding, alpha, action):
        action_hat = self.forward(state, language_embedding, alpha, action)
        return F.mse_loss(action_hat, action)

    def config(self):
        return {
            "model_arch_version": MODEL_ARCH_VERSION,
            "state_dim": self.state_dim,
            "language_dim": self.language_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        }

    def save_bundle(self, run_dir, language_index_path=None, extra_config=None):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), run_dir / "model.pt")
        config = self.config()
        if language_index_path is not None:
            config["language_index_path"] = str(language_index_path)
        if extra_config is not None:
            config.update(extra_config)
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_bundle(cls, run_dir, map_location="cpu"):
        run_dir = Path(run_dir)
        with (run_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(
            state_dim    = config["state_dim"],
            language_dim = config.get("language_dim", LANGUAGE_DIM),
            action_dim   = config.get("action_dim", ACTION_DIM),
            latent_dim   = config.get("latent_dim", LATENT_DIM),
            hidden_dim   = config.get("hidden_dim", HIDDEN_DIM),
        )
        state_dict = torch.load(run_dir / "model.pt", map_location=map_location)
        compressor_weight = state_dict.get("compressor.0.weight")
        expected_in_dim = model.hidden_dim + model.action_dim
        if compressor_weight is not None and compressor_weight.shape[1] != expected_in_dim:
            raise RuntimeError(
                "Incompatible LILAC checkpoint at %s. "
                "This model was trained with the old action-only compressor "
                "(compressor.0.weight shape=%s), but the current code expects "
                "the state/language-conditioned compressor shape [hidden_dim, hidden_dim + action_dim] "
                "with input dim %d. Restart the notebook kernel and rerun 02_lilac_train_sh5.ipynb "
                "from the first cell to produce a new run with latent_alignment.npz."
                % (run_dir, tuple(compressor_weight.shape), expected_in_dim)
            )
        model.load_state_dict(state_dict)
        model.eval()
        return model, config


class LILAWithoutCorrections(LILACModel):
    """
    LILA baseline: same latent-action model with alpha fixed to 1.
    """

    def decoder(self, state, language_embedding, alpha, z):
        alpha = torch.ones((state.shape[0],), dtype=state.dtype, device=state.device)
        return super().decoder(state, language_embedding, alpha, z)

    def forward(self, state, language_embedding, alpha, action):
        alpha = torch.ones((state.shape[0],), dtype=state.dtype, device=state.device)
        return super().forward(state, language_embedding, alpha, action)
