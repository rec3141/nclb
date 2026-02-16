"""Fit score function and community metrics.

Fit score measures how well a contig belongs in a bin.
Community metrics measure the collective coherence of a bin.
These aren't metaphors — they're concrete, differentiable metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr

from .identity import ContigIdentity, CommunityProfile


# ---------------------------------------------------------------------------
# Contig fit score: how well does this contig belong here?
# ---------------------------------------------------------------------------

def contig_fit_score(
    contig: ContigIdentity,
    community: CommunityProfile,
    adjacency: Optional[dict[str, list[str]]] = None,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute how well a contig fits in a community.

    Returns a continuous score in [-1, +1].

    When a signal has insufficient data (e.g. no coverage variation, no graph
    edges), its weight is redistributed proportionally to the available signals
    so the score isn't penalized for missing information.
    """
    base_weights = weights or {
        "tnf": 0.30,        # composition is fundamental
        "coverage": 0.25,   # abundance pattern is strong signal
        "graph": 0.15,      # graph links are physical evidence
        "recognition": 0.15,  # binner consensus carries weight
        "contribution": 0.15,  # filling gaps is valued
    }

    # --- Compute each signal, tracking which have real data ---
    signals = {}  # name → value (0-1)
    available = {}  # name → True if signal has data

    # TNF similarity: cosine similarity of contig TNF to community centroid
    tnf_sim = 0.0
    has_tnf = community.tnf_centroid is not None and contig.tnf is not None
    if has_tnf:
        cos_dist = cosine_distance(contig.tnf, community.tnf_centroid)
        tnf_sim = 1.0 - cos_dist
    signals["tnf"] = tnf_sim
    available["tnf"] = has_tnf

    # Coverage correlation: Pearson correlation of coverage profiles
    cov_corr = 0.0
    has_cov = False
    if community.mean_coverage is not None and contig.coverage is not None:
        if len(contig.coverage) > 1 and np.std(contig.coverage) > 0 and np.std(community.mean_coverage) > 0:
            r, _ = pearsonr(contig.coverage, community.mean_coverage)
            cov_corr = max(0.0, r)
            has_cov = True
        elif len(contig.coverage) == 1:
            if community.mean_coverage[0] > 0:
                ratio = contig.coverage[0] / community.mean_coverage[0]
                cov_corr = 1.0 - min(abs(np.log2(max(ratio, 0.01))), 3.0) / 3.0
                cov_corr = max(0.0, cov_corr)
                has_cov = True
    signals["coverage"] = cov_corr
    available["coverage"] = has_cov

    # Graph fraction: fraction of contig's graph neighbors in this community
    graph_frac = 0.0
    has_graph = adjacency is not None and len(contig.connections) > 0
    if has_graph:
        member_set = set(community.members)
        n_in_community = sum(1 for n in contig.connections if n in member_set)
        graph_frac = n_in_community / len(contig.connections)
    signals["graph"] = graph_frac
    available["graph"] = has_graph

    # Recognition: fraction of binners that agree with this community
    recognition = 0.0
    has_recog = bool(contig.binner_assignments)
    if has_recog:
        n_agree = 0
        for binner, assignment in contig.binner_assignments.items():
            if assignment is not None:
                if community.source_binner in assignment.lower():
                    if assignment in community.name:
                        n_agree += 1
                elif assignment is not None:
                    n_agree += 0.2
        total_binners = max(len(contig.binner_assignments), 1)
        recognition = min(n_agree / total_binners, 1.0)
    signals["recognition"] = recognition
    available["recognition"] = has_recog

    # Contribution: does contig carry genes the community is missing?
    contribution = 0.0
    has_contrib = bool(contig.marker_genes and community.missing_markers)
    if has_contrib:
        overlap = set(contig.marker_genes) & set(community.missing_markers)
        contribution = len(overlap) / len(community.missing_markers)
    signals["contribution"] = contribution
    available["contribution"] = has_contrib

    # --- Redistribute unavailable weights proportionally ---
    active_weight = sum(base_weights[k] for k in base_weights if available.get(k))
    if active_weight <= 0:
        return -1.0  # no signals at all

    raw_score = 0.0
    for name in base_weights:
        if available.get(name):
            w = base_weights[name] / active_weight  # normalized weight
            raw_score += w * signals[name]

    # Rescale to [-1, +1]
    return 2.0 * raw_score - 1.0


# ---------------------------------------------------------------------------
# Community metrics: collective coherence
# ---------------------------------------------------------------------------

def community_metrics(
    community: CommunityProfile,
    identities: dict[str, ContigIdentity],
    adjacency: Optional[dict[str, list[str]]] = None,
) -> dict[str, float]:
    """Compute collective fit metrics for a community."""

    member_scores = []
    for name in community.members:
        contig = identities.get(name)
        if contig:
            v = contig_fit_score(contig, community, adjacency)
            member_scores.append(v)

    if not member_scores:
        return {
            "mean_fit": 0.0,
            "min_fit": 0.0,
            "wholeness": community.completeness,
            "purity": 100.0 - community.redundancy,
            "collective_fit": 0.0,
            "n_uneasy": 0,
        }

    mean_v = float(np.mean(member_scores))
    min_v = float(np.min(member_scores))

    return {
        "mean_fit": mean_v,
        "min_fit": min_v,
        "wholeness": community.completeness,
        "purity": 100.0 - community.redundancy,
        "collective_fit": mean_v * (1.0 - community.redundancy / 100.0),
        "n_uneasy": sum(1 for v in member_scores if v < 0.0),
    }


# ---------------------------------------------------------------------------
# TNF coherence: how well do compositions blend?
# ---------------------------------------------------------------------------

def tnf_coherence(
    members: list[ContigIdentity],
) -> tuple[float, float]:
    """Mean and stdev of cosine similarity of each member's TNF to the centroid.

    Returns (mean_similarity, stdev_similarity).
    """
    if len(members) < 2:
        return 1.0, 0.0

    tnf_matrix = np.array([c.tnf for c in members])
    centroid = tnf_matrix.mean(axis=0)

    similarities = []
    for c in members:
        sim = 1.0 - cosine_distance(c.tnf, centroid)
        similarities.append(sim)

    return float(np.mean(similarities)), float(np.std(similarities))


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
