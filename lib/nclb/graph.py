"""Assembly graph parsing and connectivity metrics.

Social bonds between contigs, derived from the Flye assembly graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .identity import ContigIdentity, CommunityProfile


def graph_connectivity(members: list[str], adjacency: dict[str, list[str]]) -> float:
    """Fraction of member pairs connected by at least one graph edge.

    0.0 = no members share graph edges (disconnected)
    1.0 = every member pair is directly linked
    """
    if len(members) < 2:
        return 1.0  # singleton or empty is trivially connected

    member_set = set(members)
    n_pairs = 0
    n_connected = 0

    for i, a in enumerate(members):
        neighbors_a = set(adjacency.get(a, []))
        for b in members[i + 1:]:
            n_pairs += 1
            if b in neighbors_a:
                n_connected += 1

    return n_connected / n_pairs if n_pairs > 0 else 0.0


def find_graph_bridges(
    contig: str,
    communities: dict[str, CommunityProfile],
    adjacency: dict[str, list[str]],
) -> dict[str, int]:
    """Find which communities a contig is graph-connected to.

    Returns: {community_name: n_edges_to_community}
    """
    neighbors = adjacency.get(contig, [])
    community_edges: dict[str, int] = {}

    # Build reverse lookup: contig → community
    contig_to_community = {}
    for comm in communities.values():
        for member in comm.members:
            contig_to_community[member] = comm.name

    for neighbor in neighbors:
        comm = contig_to_community.get(neighbor)
        if comm:
            community_edges[comm] = community_edges.get(comm, 0) + 1

    return community_edges


def shared_edge_communities(
    communities: dict[str, CommunityProfile],
    adjacency: dict[str, list[str]],
) -> list[tuple[str, str, int]]:
    """Find pairs of communities connected by assembly graph edges.

    Returns: [(community_a, community_b, n_cross_edges)]
    """
    contig_to_community = {}
    for comm in communities.values():
        for member in comm.members:
            contig_to_community[member] = comm.name

    cross_edges: dict[tuple[str, str], int] = {}

    for contig, neighbors in adjacency.items():
        comm_a = contig_to_community.get(contig)
        if not comm_a:
            continue
        for neighbor in neighbors:
            comm_b = contig_to_community.get(neighbor)
            if comm_b and comm_b != comm_a:
                pair = tuple(sorted([comm_a, comm_b]))
                cross_edges[pair] = cross_edges.get(pair, 0) + 1

    return [(a, b, n) for (a, b), n in sorted(cross_edges.items(), key=lambda x: -x[1])]
