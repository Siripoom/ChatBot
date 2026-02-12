import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


class QueryNormalizer:
    """Normalize query text and apply configurable synonym aliases."""

    _warned_synonym_load_failure = False

    def __init__(self, synonym_path: str = "./data/query_synonyms.json"):
        self.synonym_path = Path(synonym_path)
        self._alias_pairs: List[Tuple[str, str]] = []
        self._canonicals: List[str] = []
        self._load_aliases()

    @staticmethod
    def _normalize_basic(text: str) -> str:
        lowered = text.strip().lower()
        return re.sub(r"\s+", " ", lowered)

    def normalize_query(self, text: str) -> Dict[str, Any]:
        normalized_text = self._normalize_basic(text)
        if not normalized_text:
            return {"normalized_text": "", "matched_canonicals": []}

        matched_canonicals: List[str] = []

        for alias, canonical in self._alias_pairs:
            if alias in normalized_text:
                normalized_text = normalized_text.replace(alias, canonical)
                if canonical not in matched_canonicals:
                    matched_canonicals.append(canonical)

        normalized_text = self._normalize_basic(normalized_text)

        for canonical in self._canonicals:
            if canonical in normalized_text and canonical not in matched_canonicals:
                matched_canonicals.append(canonical)

        return {
            "normalized_text": normalized_text,
            "matched_canonicals": matched_canonicals,
        }

    def _load_aliases(self) -> None:
        self._alias_pairs = []
        self._canonicals = []

        if not self.synonym_path.exists():
            self._warn_once(
                f"⚠️ Synonym file not found: {self.synonym_path} (fallback to default normalization)"
            )
            return

        try:
            with self.synonym_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # pragma: no cover
            self._warn_once(
                f"⚠️ Failed to load synonym file {self.synonym_path}: {exc} "
                "(fallback to default normalization)"
            )
            return

        aliases = payload.get("aliases", []) if isinstance(payload, dict) else []
        alias_pairs: List[Tuple[str, str]] = []
        canonicals: List[str] = []

        for entry in aliases:
            if not isinstance(entry, dict):
                continue
            canonical_raw = str(entry.get("canonical", "")).strip()
            if not canonical_raw:
                continue

            canonical = self._normalize_basic(canonical_raw)
            if not canonical:
                continue
            if canonical not in canonicals:
                canonicals.append(canonical)

            terms = entry.get("terms", [])
            if not isinstance(terms, list):
                continue

            for term in terms:
                alias = self._normalize_basic(str(term))
                if not alias or alias == canonical:
                    continue
                alias_pairs.append((alias, canonical))

        alias_pairs.sort(key=lambda item: len(item[0]), reverse=True)
        self._alias_pairs = alias_pairs
        self._canonicals = sorted(canonicals, key=len, reverse=True)

    @classmethod
    def _warn_once(cls, message: str) -> None:
        if cls._warned_synonym_load_failure:
            return
        print(message)
        cls._warned_synonym_load_failure = True
