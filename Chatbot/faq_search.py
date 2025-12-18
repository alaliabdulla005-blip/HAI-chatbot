"""
FAQ search module.

- Loads FAQ entries from faqs.csv via data_loader.
- Builds a TF–IDF index over questions (and optionally answers).
- Supports similarity-based lookup with optional tag filters.

This is a simple IR component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .data_loader import load_faqs
from .domain_models import FAQEntry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAQS_CSV_PATH = PROJECT_ROOT / "data" / "faqs.csv"


@dataclass
class FAQMatch:
    """Result of a FAQ search."""
    faq: FAQEntry
    score: float  # cosine similarity in [0, 1]


class FAQSearchEngine:
    """
    Simple FAQ search over faqs.csv using TF–IDF + cosine similarity.

    Usage:
        engine = FAQSearchEngine.from_csv()
        matches = engine.search("Is Misto oat dairy free?",
                                top_k=3,
                                min_score=0.2,
                                required_tags={"allergens"})
        if matches:
            print(matches[0].faq.answer_markdown)
    """

    def __init__(
        self,
        faqs: Sequence[FAQEntry],
        vectorizer: TfidfVectorizer,
        doc_matrix,
    ) -> None:
        self._faqs: List[FAQEntry] = list(faqs)
        self._vectorizer = vectorizer
        self._doc_matrix = doc_matrix  # sparse matrix (n_faqs, n_features)
        # Convenience index for direct lookup by ID
        self._by_id = {faq.faq_id: faq for faq in self._faqs}

    # ---------- Construction helpers ----------

    @classmethod
    def from_entries(
        cls,
        faqs: Sequence[FAQEntry],
        use_answers: bool = True,
    ) -> "FAQSearchEngine":
        """
        Build a fresh TF–IDF index from FAQ entries.

        :param faqs: list of FAQEntry objects.
        :param use_answers: if True, index question + answer_markdown;
                            else index only the question text.
        """
        texts: List[str] = []
        for faq in faqs:
            base = faq.question or ""
            if use_answers and faq.answer_markdown:
                combined = f"{base} {faq.answer_markdown}"
            else:
                combined = base
            texts.append(combined)

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True,
            strip_accents="unicode",
            # default norm="l2" makes dot-product ~= cosine similarity
        )
        doc_matrix = vectorizer.fit_transform(texts)

        return cls(faqs=faqs, vectorizer=vectorizer, doc_matrix=doc_matrix)

    @classmethod
    def from_csv(
        cls,
        csv_path: Optional[Path] = None,
        use_answers: bool = True,
    ) -> "FAQSearchEngine":
        """
        Load faqs.csv and build an index.

        :param csv_path: optional override for faqs.csv path.
        """
        if csv_path is None:
            csv_path = FAQS_CSV_PATH
        faqs = load_faqs(csv_path)
        if not faqs:
            raise ValueError(f"No FAQ entries found in {csv_path}")
        return cls.from_entries(faqs, use_answers=use_answers)

    # ---------- Search API ----------

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.2,
        required_tags: Optional[Iterable[str]] = None,
    ) -> List[FAQMatch]:
        """
        Rank FAQs by similarity to a user query.

        :param query: user text.
        :param top_k: maximum number of matches to return.
        :param min_score: minimum cosine similarity to accept a result.
        :param required_tags: if provided, only FAQs whose tags intersect
                              this set are considered (e.g. {"allergens"}).
        :return: list of FAQMatch ordered by descending score.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return []

        required: Set[str] = set(required_tags or [])

        q_vec = self._vectorizer.transform([cleaned])
        # linear_kernel works as cosine similarity when vectors are l2-normalised
        sims: np.ndarray = linear_kernel(q_vec, self._doc_matrix)[0]

        # Optionally mask out FAQs whose tags do not match required_tags
        if required:
            mask = np.ones_like(sims, dtype=bool)
            for i, faq in enumerate(self._faqs):
                raw = faq.tags or ""
                faq_tags = {t.strip() for t in raw.split("|") if t.strip()}
                if faq_tags.isdisjoint(required):
                    mask[i] = False
            sims = np.where(mask, sims, 0.0)

        # Rank indices by similarity (high → low)
        indices = np.argsort(-sims)

        matches: List[FAQMatch] = []
        for idx in indices:
            score = float(sims[idx])
            if score < min_score:
                break
            matches.append(FAQMatch(faq=self._faqs[idx], score=score))
            if len(matches) >= top_k:
                break

        return matches

    # ---------- Utility methods ----------

    def get_faq_by_id(self, faq_id: str) -> Optional[FAQEntry]:
        """Direct lookup of a FAQ entry by its ID."""
        return self._by_id.get(faq_id)

    @property
    def faqs(self) -> Sequence[FAQEntry]:
        """Expose the underlying FAQ list (read-only)."""
        return tuple(self._faqs)
