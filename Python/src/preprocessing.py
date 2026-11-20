"""
Single preprocessing path for training, evaluation and runtime.

Every place that turns UNSW-NB15-style data into model input must go through
FeaturePipeline. That covers train_models_v2.ipynb, evaluations_v2.ipynb and
detector.py. Nothing else may encode categoricals or call the scaler.

Contract (feature_meta.json, schema_version 2):
  feature_cols : ordered list of the 38 model inputs
  cat_cols     : categorical columns (proto, state)
  categories   : per categorical, the ordered list of string LABELS seen in the
                 UNSW-NB15 *training* partition. A value's code is its index.
  unknown_code : code used for a label not in the vocabulary (-1.0)

Schema 1 (written by train_models.ipynb, v1) stored integer codes instead of
labels. That made runtime encoding impossible, so schema 1 is rejected.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

SCHEMA_VERSION = 2
UNKNOWN_CODE = -1.0

FEATURE_COLS = [
    "dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sjit", "djit", "swin", "dwin", "stcpb",
    "dtcpb", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_src_ltm",
    "ct_srv_dst", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "is_sm_ips_ports",
]
CAT_COLS = ["proto", "state"]


class SchemaError(RuntimeError):
    pass


class FeaturePipeline:
    def __init__(self, feature_cols: Iterable[str], cat_cols: Iterable[str],
                 vocab: Mapping[str, Iterable[str]], scaler=None):
        self.feature_cols = list(feature_cols)
        self.cat_cols = list(cat_cols)
        self.vocab = {c: [str(v) for v in vocab[c]] for c in self.cat_cols}
        self.scaler = scaler

        missing = [c for c in self.cat_cols if c not in self.feature_cols]
        if missing:
            raise SchemaError(f"cat_cols not in feature_cols: {missing}")

        self._index = {c: {v: i for i, v in enumerate(vals)} for c, vals in self.vocab.items()}
        # Case-insensitive fallback (runtime sends lower-case labels). Refuse
        # vocabularies where lower-casing would merge two labels.
        self._index_ci = {}
        for c, vals in self.vocab.items():
            ci = {}
            for i, v in enumerate(vals):
                k = v.lower()
                if k in ci:
                    raise SchemaError(f"{c}: labels collide case-insensitively: {vals[ci[k]]!r} / {v!r}")
                ci[k] = i
            self._index_ci[c] = ci

        if scaler is not None and getattr(scaler, "n_features_in_", len(self.feature_cols)) != len(self.feature_cols):
            raise SchemaError(f"scaler expects {scaler.n_features_in_} features, contract has {len(self.feature_cols)}")

    # ---------- construction / persistence ----------
    @classmethod
    def fit_vocab(cls, train_df: pd.DataFrame, feature_cols=FEATURE_COLS, cat_cols=CAT_COLS,
                  scaler=None) -> "FeaturePipeline":
        """Vocabulary from the TRAINING partition only, sorted (same order a
        LabelEncoder / pandas Categorical would use)."""
        vocab = {c: sorted(train_df[c].astype(str).unique().tolist()) for c in cat_cols}
        return cls(feature_cols, cat_cols, vocab, scaler)

    def to_meta(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "categories": self.vocab,
            "unknown_code": UNKNOWN_CODE,
            "vocab_source": "UNSW-NB15 training partition, sorted labels",
        }

    def save_meta(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_meta(), f, indent=2)

    @staticmethod
    def read_meta(path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("schema_version") != SCHEMA_VERSION:
            cats = meta.get("categories", {})
            all_ints = all(isinstance(v, int) for vals in cats.values() for v in vals)
            hint = (" Its 'categories' hold integer codes, not labels (v1 bug), so no "
                    "string can be encoded." if all_ints and cats else "")
            raise SchemaError(f"{path}: schema_version {meta.get('schema_version')!r}, need "
                              f"{SCHEMA_VERSION}.{hint} Regenerate it with train_models_v2.ipynb.")
        return meta

    @classmethod
    def load(cls, model_dir) -> "FeaturePipeline":
        model_dir = Path(model_dir)
        meta = cls.read_meta(model_dir / "feature_meta.json")
        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return cls(meta["feature_cols"], meta["cat_cols"], meta["categories"], scaler)

    # ---------- encoding ----------
    def _code(self, col: str, value) -> float:
        s = "" if value is None else str(value)
        i = self._index[col].get(s)
        if i is None:
            i = self._index_ci[col].get(s.lower())
        return UNKNOWN_CODE if i is None else float(i)

    def encode_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """Raw (unscaled) n x 38 matrix + count of unknown labels per categorical."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise SchemaError(f"input frame lacks feature columns: {missing}")
        out = np.empty((len(df), len(self.feature_cols)), dtype=float)
        unknown = {}
        for j, c in enumerate(self.feature_cols):
            if c in self.cat_cols:
                codes = df[c].map(lambda v, c=c: self._code(c, v)).to_numpy(dtype=float)
                unknown[c] = int((codes == UNKNOWN_CODE).sum())
                out[:, j] = codes
            else:
                out[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), unknown

    def encode_record(self, feats: Mapping) -> tuple[np.ndarray, list]:
        """Raw (unscaled) 1 x 38 vector for one live record + list of unknown cat cols."""
        vec, unknown = [], []
        for c in self.feature_cols:
            if c in self.cat_cols:
                code = self._code(c, feats.get(c, ""))
                if code == UNKNOWN_CODE:
                    unknown.append(c)
                vec.append(code)
            else:
                try:
                    vec.append(float(feats.get(c, 0.0)))
                except (TypeError, ValueError):
                    vec.append(0.0)
        x = np.nan_to_num(np.asarray([vec], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        return x, unknown

    # ---------- scaling ----------
    def scale(self, X_raw: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise SchemaError("no scaler attached; load the pipeline with FeaturePipeline.load()")
        return self.scaler.transform(np.asarray(X_raw, dtype=float))

    def prepare_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        X, unknown = self.encode_frame(df)
        return self.scale(X), unknown

    def prepare_record(self, feats: Mapping) -> tuple[np.ndarray, list]:
        x, unknown = self.encode_record(feats)
        return self.scale(x), unknown