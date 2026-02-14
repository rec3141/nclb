"""Valence function and harmony metrics.

Valence measures how well a contig belongs in a community.
Harmony measures the collective wellbeing of a community.
These aren't metaphors — they're concrete, differentiable metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr

from .identity import ContigIdentity, CommunityProfile


# ---------------------------------------------------------------------------
# Contig valence: how well does this contig belong here?
# ---------------------------------------------------------------------------

def contig_valence(
    contig: ContigIdentity,
    community: CommunityProfile,
    adjacency: Optional[dict[str, list[str]]] = None,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute how well a contig resonates with a community.

    Returns a continuous score in [-1, +1].
    """
    w = weights or {
        "harmony": 0.30,    # composition is fundamental
        "rhythm": 0.25,     # abundance pattern is strong signal
        "kinship": 0.15,    # graph links are physical evidence
        "recognition": 0.15,  # binner consensus carries weight
        "contribution": 0.15,  # filling gaps is valued
    }

    # Harmony: cosine similarity of contig TNF to community centroid
    harmony = 0.0
    if community.tnf_centroid is not None and contig.tnf is not None:
        cos_dist = cosine_distance(contig.tnf, community.tnf_centroid)
        harmony = 1.0 - cos_dist  # cosine similarity

    # Rhythm: Pearson correlation of coverage profiles
    rhythm = 0.0
    if community.mean_coverage is not None and contig.coverage is not None:
        if len(contig.coverage) > 1 and np.std(contig.coverage) > 0 and np.std(community.mean_coverage) > 0:
            r, _ = pearsonr(contig.coverage, community.mean_coverage)
            rhythm = max(0.0, r)  # clamp negative correlations to 0
        elif len(contig.coverage) == 1:
            # Single sample — use ratio similarity instead
            if community.mean_coverage[0] > 0:
                ratio = contig.coverage[0] / community.mean_coverage[0]
                rhythm = 1.0 - min(abs(np.log2(max(ratio, 0.01))), 3.0) / 3.0
                rhythm = max(0.0, rhythm)

    # Kinship: fraction of contig's graph neighbors in this community
    kinship = 0.0
    if adjacency is not None and contig.connections:
        member_set = set(community.members)
        n_in_community = sum(1 for n in contig.connections if n in member_set)
        kinship = n_in_community / len(contig.connections)

    # Recognition: fraction of oracles that agree with this community
    recognition = 0.0
    if contig.testimony:
        # For each binner, check if it placed this contig in the same bin
        # that was the source of this community
        n_agree = 0
        for binner, assignment in contig.testimony.items():
            if assignment is not None:
                # Check if this binner's assignment matches the community's source
                # community name format: dastool-{binner}_{number}
                if community.source_binner in assignment.lower():
                    # Same binner — check if it's the same bin number
                    if assignment in community.name:
                        n_agree += 1
                elif assignment is not None:
                    # Different binner — count as partial agreement if any assignment
                    n_agree += 0.2
        recognition = min(n_agree / 5.0, 1.0)

    # Contribution: does contig carry genes the community is missing?
    contribution = 0.0
    if contig.marker_genes and community.missing_markers:
        overlap = set(contig.marker_genes) & set(community.missing_markers)
        contribution = len(overlap) / len(community.missing_markers)

    # Weighted combination
    raw_valence = (
        w["harmony"] * harmony
        + w["rhythm"] * rhythm
        + w["kinship"] * kinship
        + w["recognition"] * recognition
        + w["contribution"] * contribution
    )

    # Rescale to [-1, +1]
    return 2.0 * raw_valence - 1.0


# ---------------------------------------------------------------------------
# Community harmony: collective wellbeing
# ---------------------------------------------------------------------------

def community_harmony(
    community: CommunityProfile,
    identities: dict[str, ContigIdentity],
    adjacency: Optional[dict[str, list[str]]] = None,
) -> dict[str, float]:
    """Compute collective harmony metrics for a community."""

    member_valences = []
    for name in community.members:
        contig = identities.get(name)
        if contig:
            v = contig_valence(contig, community, adjacency)
            member_valences.append(v)

    if not member_valences:
        return {
            "mean_valence": 0.0,
            "min_valence": 0.0,
            "wholeness": community.completeness,
            "purity": 100.0 - community.redundancy,
            "collective_harmony": 0.0,
            "n_uneasy": 0,
        }

    mean_v = float(np.mean(member_valences))
    min_v = float(np.min(member_valences))

    return {
        "mean_valence": mean_v,
        "min_valence": min_v,
        "wholeness": community.completeness,
        "purity": 100.0 - community.redundancy,
        "collective_harmony": mean_v * (1.0 - community.redundancy / 100.0),
        "n_uneasy": sum(1 for v in member_valences if v < 0.0),
    }


# ---------------------------------------------------------------------------
# TNF coherence: how well do compositions blend?
# ---------------------------------------------------------------------------

def tnf_coherence(
    members: list[ContigIdentity],
) -> float:
    """Mean cosine similarity of each member's TNF to the centroid."""
    if len(members) < 2:
        return 1.0

    tnf_matrix = np.array([c.tnf for c in members])
    centroid = tnf_matrix.mean(axis=0)

    similarities = []
    for c in members:
        sim = 1.0 - cosine_distance(c.tnf, centroid)
        similarities.append(sim)

    return float(np.mean(similarities))


def coverage_coherence(
    members: list[ContigIdentity],
) -> float:
    """Mean pairwise Pearson correlation of coverage profiles."""
    if len(members) < 2:
        return 1.0

    correlations = []
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if np.std(a.coverage) > 0 and np.std(b.coverage) > 0:
                r, _ = pearsonr(a.coverage, b.coverage)
                correlations.append(r)

    return float(np.mean(correlations)) if correlations else 0.0
