"""Resonance maps — who is like me?

KNN indices for finding contigs that resonate by composition and energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

from .identity import ContigIdentity, CommunityProfile


@dataclass
class ResonanceCandidate:
    """An unbinned contig that resonates with a community."""
    contig_name: str
    community_name: str
    tnf_similarity: float       # cosine similarity to community centroid
    coverage_correlation: float  # Pearson r with community mean coverage
    graph_edges: int             # number of graph edges to community members
    voice_agreement: int         # oracles that placed contig near this community
    score: float                 # composite resonance score


class ResonanceMap:
    """Pre-computed nearest-neighbor indices for fast resonance queries."""

    def __init__(
        self,
        identities: dict[str, ContigIdentity],
        communities: dict[str, CommunityProfile],
        adjacency: dict[str, list[str]],
        k: int = 20,
    ):
        self.identities = identities
        self.communities = communities
        self.adjacency = adjacency
        self.k = k

        # Build contig name → index mapping
        self.contig_names = sorted(identities.keys())
        self.name_to_idx = {n: i for i, n in enumerate(self.contig_names)}

        # Build feature matrices
        self.tnf_matrix = np.array([identities[n].tnf for n in self.contig_names])
        self.cov_matrix = np.array([identities[n].coverage for n in self.contig_names])

        # Normalize TNF for cosine KNN (sklearn uses Euclidean, so we L2-normalize)
        norms = np.linalg.norm(self.tnf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tnf_normalized = self.tnf_matrix / norms

        # Build KNN index on normalized TNF
        self._tnf_knn = NearestNeighbors(n_neighbors=min(k, len(self.contig_names)), metric="euclidean")
        self._tnf_knn.fit(self.tnf_normalized)

    def find_resonant_contigs(
        self,
        community: CommunityProfile,
        max_candidates: int = 50,
        min_score: float = 0.3,
    ) -> list[ResonanceCandidate]:
        """Find unbinned contigs that resonate with a community.

        Uses TNF KNN to find compositionally similar contigs, then
        scores them by multi-signal concordance.
        """
        if community.tnf_centroid is None:
            return []

        # Normalize centroid
        centroid = community.tnf_centroid
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return []
        centroid_norm = centroid / norm

        # Find K nearest neighbors to centroid in TNF space
        distances, indices = self._tnf_knn.kneighbors(
            centroid_norm.reshape(1, -1),
            n_neighbors=min(self.k * 5, len(self.contig_names)),
        )

        member_set = set(community.members)
        candidates = []

        for dist, idx in zip(distances[0], indices[0]):
            name = self.contig_names[idx]
            contig = self.identities[name]

            # Skip if already in a community
            if contig.community is not None:
                continue

            # TNF similarity (convert L2 distance on normalized vectors to cosine sim)
            tnf_sim = 1.0 - cosine_distance(contig.tnf, centroid)

            # Coverage correlation
            cov_corr = 0.0
            if community.mean_coverage is not None:
                if np.std(contig.coverage) > 0 and np.std(community.mean_coverage) > 0:
                    r, _ = pearsonr(contig.coverage, community.mean_coverage)
                    cov_corr = max(0.0, r)

            # Graph edges to community
            neighbors = set(self.adjacency.get(name, []))
            graph_edges = len(neighbors & member_set)

            # Voice agreement: oracles that placed contig in a bin from the same binner
            voice_agreement = 0
            for binner, assignment in contig.testimony.items():
                if assignment is not None and community.source_binner in assignment:
                    voice_agreement += 1

            # Composite score
            score = (
                0.30 * tnf_sim
                + 0.30 * cov_corr
                + 0.20 * min(graph_edges / 3.0, 1.0)
                + 0.20 * (voice_agreement / 5.0)
            )

            if score >= min_score:
                candidates.append(ResonanceCandidate(
                    contig_name=name,
                    community_name=community.name,
                    tnf_similarity=tnf_sim,
                    coverage_correlation=cov_corr,
                    graph_edges=graph_edges,
                    voice_agreement=voice_agreement,
                    score=score,
                ))

        # Sort by score descending
        candidates.sort(key=lambda c: -c.score)
        return candidates[:max_candidates]

    def find_nearest_contigs(
        self,
        contig_name: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find K contigs most similar to a given contig by TNF.

        Returns: [(contig_name, cosine_similarity)]
        """
        idx = self.name_to_idx.get(contig_name)
        if idx is None:
            return []

        query = self.tnf_normalized[idx].reshape(1, -1)
        distances, indices = self._tnf_knn.kneighbors(query, n_neighbors=min(k + 1, len(self.contig_names)))

        results = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            neighbor_name = self.contig_names[neighbor_idx]
            if neighbor_name == contig_name:
                continue
            sim = 1.0 - cosine_distance(
                self.identities[contig_name].tnf,
                self.identities[neighbor_name].tnf,
            )
            results.append((neighbor_name, sim))

        return results[:k]
