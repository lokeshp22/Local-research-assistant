"""Similarity helpers built on nomic-embed-text vectors.

Two consumers:
  1. the planner, to merge near-duplicate sub-questions before dispatch
     (cosine > DEDUP_THRESHOLD);
  2. the critic, to find claim pairs similar enough to be *about* the same
     thing, which is the precondition for them being able to contradict.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


_WORD_RE = __import__("re").compile(r"[a-z][a-z'-]+")
_STOP = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "what", "how", "why", "does", "did", "has", "have", "its", "their", "been",
    "into", "than", "then", "they", "them", "there", "which", "about", "over",
    "also", "such", "some", "more", "most", "other", "been", "being", "only",
    "according", "report", "reported", "says", "said", "source",
}


def content_tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def lexical_overlap(a: str, b: str) -> float:
    """Overlap coefficient over content words: |A∩B| / min(|A|,|B|).

    Used alongside cosine similarity because embedding models frequently place
    a statement and its negation *far apart* — "capacity rose 14%" and
    "capacity fell 6%" can sit below any sensible cosine threshold while being
    exactly the pair the critic needs to see. Shared vocabulary catches those.
    """
    ta, tb = content_tokens(a), content_tokens(b)
    if len(ta) < 4 or len(tb) < 4:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def to_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = normalize_rows(to_matrix(vectors))
    if matrix.size == 0:
        return matrix
    return matrix @ matrix.T


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def similar_pairs(
    vectors: Sequence[Sequence[float]], threshold: float
) -> list[tuple[int, int, float]]:
    """All (i, j, similarity) with i < j above threshold, highest first."""
    sim = cosine_matrix(vectors)
    if sim.size == 0:
        return []
    pairs: list[tuple[int, int, float]] = []
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim[i, j])
            if score >= threshold:
                pairs.append((i, j, round(score, 4)))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def cluster(n: int, pairs: Iterable[tuple[int, int, float]]) -> list[list[int]]:
    """Union-find over similar pairs. Returns clusters in original index order."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)
    return [sorted(members) for _, members in sorted(groups.items())]


def merge_subquestions(
    subquestions: Sequence[str],
    vectors: Sequence[Sequence[float]],
    threshold: float,
) -> tuple[list[str], list[dict]]:
    """Collapse near-duplicate sub-questions.

    The representative kept from each cluster is the longest one, on the theory
    that the more specific phrasing carries more retrievable signal. Returns
    (merged_subquestions, merge_notes) where each note records what was folded
    into what and at what similarity — the run log needs this.
    """
    if len(subquestions) < 2 or len(vectors) != len(subquestions):
        return list(subquestions), []

    pairs = similar_pairs(vectors, threshold)
    if not pairs:
        return list(subquestions), []

    sim = cosine_matrix(vectors)
    clusters = cluster(len(subquestions), pairs)

    merged: list[str] = []
    notes: list[dict] = []
    for members in clusters:
        if len(members) == 1:
            merged.append(subquestions[members[0]])
            continue
        keeper = max(members, key=lambda idx: (len(subquestions[idx]), -idx))
        merged.append(subquestions[keeper])
        notes.append(
            {
                "kept": subquestions[keeper],
                "merged_away": [subquestions[idx] for idx in members if idx != keeper],
                "max_similarity": round(
                    max(float(sim[keeper, idx]) for idx in members if idx != keeper), 4
                ),
            }
        )
    return merged, notes
