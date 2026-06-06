"""
language.py

Canonical language dataset, LIFO correction stack, Gemini utterance selection,
and SBERT-style embedding/indexing utilities for LILAC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from pathlib import Path

import numpy as np

from constants import LANGUAGE_DIM


THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent.parent
DEFAULT_CANONICAL_LANGUAGE_PATH = PROJECT_DIR / "data" / "language" / "lilac_canonical_utterances.json"
DEFAULT_SBERT_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"


def load_project_env(path=DEFAULT_ENV_PATH):
    """
    Load simple KEY=VALUE lines from the project .env file without overwriting
    variables that are already present in the shell environment.
    """
    path = Path(path)
    if not path.exists():
        return {}

    loaded = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def normalize_utterance(utterance):
    """
    Normalize an utterance for dataset lookup.
    """
    utterance = "" if utterance is None else str(utterance)
    utterance = utterance.strip().lower()
    utterance = re.sub(r"\s+", " ", utterance)
    utterance = utterance.strip(" \t\n\r.,!?;:\"'()[]{}")
    return utterance


@dataclass(frozen=True)
class CanonicalUtterance:
    id: str
    text: str
    kind: str
    alpha: float
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload):
        required = {"id", "text", "kind", "alpha"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("Canonical utterance is missing fields: %s" % ", ".join(missing))
        aliases = tuple(str(a).strip() for a in payload.get("aliases", []) if str(a).strip())
        return cls(
            id=str(payload["id"]).strip(),
            text=str(payload["text"]).strip(),
            kind=str(payload["kind"]).strip(),
            alpha=float(payload["alpha"]),
            aliases=aliases,
        )


class CanonicalLanguageDataset:
    """
    Human-maintained canonical utterance table.

    Alpha labels and language candidates come from this file instead of online
    GPT calls or rule-based guessing.
    """

    def __init__(self, entries):
        self.entries = list(entries)
        self._by_id = {}
        self._by_text = {}
        self._validate()

    @classmethod
    def load(cls, path=DEFAULT_CANONICAL_LANGUAGE_PATH):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            raw_entries = payload
        else:
            raw_entries = payload.get("utterances")

        if raw_entries is None:
            raise ValueError("Canonical language file must contain an 'utterances' list.")
        return cls(CanonicalUtterance.from_payload(item) for item in raw_entries)

    def _validate(self):
        if not self.entries:
            raise ValueError("Canonical language dataset is empty.")
        for entry in self.entries:
            if not entry.id:
                raise ValueError("Canonical utterance id cannot be empty.")
            if entry.id in self._by_id:
                raise ValueError("Duplicate canonical utterance id: %s" % entry.id)
            self._by_id[entry.id] = entry

            keys = [entry.text] + list(entry.aliases)
            for key in keys:
                norm = normalize_utterance(key)
                if not norm:
                    raise ValueError("Canonical utterance text/alias cannot be empty for id=%s" % entry.id)
                if norm in self._by_text:
                    raise ValueError("Duplicate canonical utterance text/alias: %s" % norm)
                self._by_text[norm] = entry

    @staticmethod
    def _matches_kind(entry, kind):
        if kind is None:
            return True
        return entry.kind == str(kind).strip()

    def entries_for_kind(self, kind=None):
        return [entry for entry in self.entries if self._matches_kind(entry, kind)]

    def ids(self, kind=None):
        return [entry.id for entry in self.entries_for_kind(kind)]

    def texts(self, kind=None):
        return [entry.text for entry in self.entries_for_kind(kind)]

    def as_prompt_choices(self, kind=None):
        return [
            {
                "id": entry.id,
                "text": entry.text,
                "kind": entry.kind,
                "aliases": list(entry.aliases),
            }
            for entry in self.entries_for_kind(kind)
        ]

    def get(self, utterance_or_id, kind=None):
        key = str(utterance_or_id).strip()
        if key in self._by_id:
            entry = self._by_id[key]
            if self._matches_kind(entry, kind):
                return entry
        norm = normalize_utterance(key)
        if norm in self._by_text:
            entry = self._by_text[norm]
            if self._matches_kind(entry, kind):
                return entry
        if kind is None:
            raise KeyError("Unknown canonical utterance: %s" % utterance_or_id)
        raise KeyError("Unknown canonical %s utterance: %s" % (kind, utterance_or_id))

    def maybe_get(self, utterance_or_id, kind=None):
        try:
            return self.get(utterance_or_id, kind=kind)
        except KeyError:
            return None

    def alpha(self, utterance_or_id):
        return float(self.get(utterance_or_id).alpha)


class DatasetAlphaLabeler:
    """
    Alpha labeler backed only by the canonical language dataset.
    """

    def __init__(self, dataset=None, path=DEFAULT_CANONICAL_LANGUAGE_PATH):
        self.dataset = dataset if dataset is not None else CanonicalLanguageDataset.load(path)

    def __call__(self, utterance):
        return self.dataset.alpha(utterance)


@dataclass
class LanguageStack:
    """
    Last-in-first-out language state used by LILAC.
    """

    instruction: str = ""
    corrections: list[str] = field(default_factory=list)

    def set_instruction(self, instruction):
        self.instruction = str(instruction)
        self.corrections = []

    def push(self, correction):
        correction = str(correction).strip()
        if correction:
            self.corrections.append(correction)

    def pop(self):
        if not self.corrections:
            return None
        return self.corrections.pop()

    def clear(self):
        self.corrections = []

    def active(self):
        if self.corrections:
            return self.corrections[-1]
        return self.instruction

    def as_dict(self):
        return {
            "instruction": self.instruction,
            "corrections": list(self.corrections),
            "active": self.active(),
        }


class GeminiUtteranceSelector:
    """
    Select one canonical utterance id for free-form user text using Gemini.
    """

    def __init__(
            self,
            dataset,
            model=None,
            api_key=None,
            client=None,
        ):
        load_project_env()
        self.dataset = dataset
        self.model = model or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.client = client

    def select(self, raw_utterance, kind=None):
        exact = self.dataset.maybe_get(raw_utterance, kind=kind)
        if exact is not None:
            return exact

        allowed_ids = self.dataset.ids(kind=kind)
        if not allowed_ids:
            raise RuntimeError("No canonical utterance candidates for kind=%s." % kind)

        if self.client is None:
            if not self.api_key:
                raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY for utterance selection.")
            try:
                from google import genai
            except Exception as exc:
                raise RuntimeError("google-genai is not available: %s" % exc) from exc
            self.client = genai.Client(api_key=self.api_key)

        prompt = self._build_prompt(raw_utterance, kind=kind)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if text is None:
                text = str(response)
            selected_id = self._parse_response(text, allowed_ids=allowed_ids)
            return self.dataset.get(selected_id, kind=kind)
        except Exception as exc:
            raise RuntimeError("Gemini utterance selection failed: %s" % exc) from exc

    def _build_prompt(self, raw_utterance, kind=None):
        choices = json.dumps(self.dataset.as_prompt_choices(kind=kind), ensure_ascii=True, indent=2)
        kind_line = ""
        if kind is not None:
            kind_line = (
                "The requested command type is '%s'. Only choose an id whose kind is '%s'.\n"
                % (kind, kind)
            )
        return (
            "You are a strict canonicalizer for robot language commands.\n"
            "You must choose the single most similar command from the allowed canonical commands.\n"
            "%s"
            "You must always choose one of the allowed ids, even if the user input "
            "has typos, extra words, or is awkwardly phrased.\n"
            "Never invent a new id. Never answer that none match.\n"
            "Return only the canonical id string and no other text.\n\n"
            "Allowed canonical commands:\n%s\n\n"
            "User input: %s\n"
            "Canonical id:" % (kind_line, choices, str(raw_utterance))
        )

    def _parse_response(self, text, allowed_ids=None):
        allowed_ids = set(self.dataset.ids() if allowed_ids is None else allowed_ids)
        cleaned = str(text).strip()
        cleaned = cleaned.strip("` \t\n\r")
        cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
        cleaned = cleaned.strip("\"'.,:; ")
        if cleaned not in allowed_ids:
            raise ValueError("Gemini returned an unknown canonical id: %s" % cleaned)
        return cleaned


class TransformerLanguageEmbedder:
    """
    Sentence-BERT style embedding wrapper using transformers only.
    """

    def __init__(
            self,
            model_name=DEFAULT_SBERT_MODEL,
            cache_dir=None,
            device="cpu",
            max_length=32,
        ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dim = LANGUAGE_DIM
        self.device = device
        self.max_length = int(max_length)
        model_name = self._resolve_cached_model_path(model_name, cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = self.model.to(device)
        self.model.eval()

    @staticmethod
    def _resolve_cached_model_path(model_name, cache_dir):
        if cache_dir is None:
            return model_name
        model_path = Path(str(model_name))
        if model_path.exists():
            return str(model_path)

        cache_root = Path(cache_dir)
        snapshot_root = cache_root / ("models--" + str(model_name).replace("/", "--"))
        ref_path = snapshot_root / "refs" / "main"
        if ref_path.exists():
            snapshot_id = ref_path.read_text(encoding="utf-8").strip()
            snapshot_path = snapshot_root / "snapshots" / snapshot_id
            if snapshot_path.exists():
                return str(snapshot_path)
        return model_name

    def encode(self, utterance):
        text = str(utterance)
        with self.torch.no_grad():
            enc = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {key: val.to(self.device) for key, val in enc.items()}
            output = self.model(**enc)
            embedding = self._sentence_pool(output, enc["attention_mask"])
            embedding = self.torch.nn.functional.normalize(embedding, dim=0)
        return embedding.detach().cpu().numpy().astype(np.float32)

    def _sentence_pool(self, output, attention_mask):
        embeddings = output[0]
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        embedding_sum = self.torch.sum(embeddings * mask, dim=1)
        mask_sum = self.torch.clamp(mask.sum(1), min=1e-9)
        return (embedding_sum / mask_sum).squeeze(0)


class CanonicalLanguageIndex:
    """
    Precomputed SBERT embeddings keyed by canonical utterance id/text.
    """

    def __init__(self, dataset, embeddings):
        self.dataset = dataset
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.shape[0] != len(self.dataset.entries):
            raise ValueError("Embedding count does not match canonical language dataset.")
        self.ids = self.dataset.ids()
        self.utterances = self.dataset.texts()
        self._idx_by_id = {canonical_id: idx for idx, canonical_id in enumerate(self.ids)}

    @classmethod
    def build(cls, dataset, embedder):
        embeddings = [embedder.encode(entry.text).reshape(-1) for entry in dataset.entries]
        return cls(dataset=dataset, embeddings=np.asarray(embeddings, dtype=np.float32))

    def lookup(self, utterance_or_id):
        entry = self.dataset.get(utterance_or_id)
        idx = self._idx_by_id[entry.id]
        return {
            "id": entry.id,
            "utterance": entry.text,
            "alpha": float(entry.alpha),
            "embedding": self.embeddings[idx].copy(),
            "idx": idx,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ids=np.asarray(self.ids, dtype=object),
            utterances=np.asarray(self.utterances, dtype=object),
            kinds=np.asarray([entry.kind for entry in self.dataset.entries], dtype=object),
            alphas=np.asarray([entry.alpha for entry in self.dataset.entries], dtype=np.float32),
            aliases=np.asarray(
                [
                    json.dumps(entry.aliases, ensure_ascii=True)
                    for entry in self.dataset.entries
                ],
                dtype=object,
            ),
            embeddings=self.embeddings,
        )
        return path

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        aliases = data["aliases"] if "aliases" in data else ["[]"] * len(data["ids"])
        entries = []
        for idx, canonical_id in enumerate(data["ids"].tolist()):
            entries.append(CanonicalUtterance(
                id=str(canonical_id),
                text=str(data["utterances"][idx]),
                kind=str(data["kinds"][idx]) if "kinds" in data else "unknown",
                alpha=float(data["alphas"][idx]) if "alphas" in data else 1.0,
                aliases=tuple(json.loads(str(aliases[idx]))),
            ))
        return cls(
            dataset=CanonicalLanguageDataset(entries),
            embeddings=np.asarray(data["embeddings"], dtype=np.float32),
        )


def load_language_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload.get("utterances", [])
