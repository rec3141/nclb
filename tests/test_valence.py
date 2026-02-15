#!/usr/bin/env python3
"""Test fit score computation on real data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from nclb.identity import build_identities, load_gfa_graph
from nclb.valence import contig_fit_score, community_metrics, tnf_coherence, coverage_coherence
from nclb.graph import graph_connectivity

RESULTS = Path("/data/danav2/20_mag_assembly/nextflow/results_lorbin_test")

identities, communities = build_identities(
    tnf_path=RESULTS / "assembly" / "tnf.tsv",
    depths_path=RESULTS / "mapping" / "depths.txt",
    assembly_info_path=RESULTS / "assembly" / "assembly_info.txt",
    gfa_path=RESULTS / "assembly" / "assembly_graph.gfa",
    binner_paths={
        "semibin": RESULTS / "binning" / "semibin" / "contig_bins.tsv",
        "metabat": RESULTS / "binning" / "metabat" / "contig_bins.tsv",
        "maxbin": RESULTS / "binning" / "maxbin" / "contig_bins.tsv",
        "lorbin": RESULTS / "binning" / "lorbin" / "contig_bins.tsv",
        "comebin": RESULTS / "binning" / "comebin" / "contig_bins.tsv",
    },
    consensus_path=RESULTS / "binning" / "dastool" / "contig2bin.tsv",
    summary_path=RESULTS / "binning" / "dastool" / "summary.tsv",
    checkm2_path=RESULTS / "binning" / "checkm2" / "quality_report.tsv",
)

adjacency = load_gfa_graph(RESULTS / "assembly" / "assembly_graph.gfa")

print("=" * 70)
print("COMMUNITY FIT REPORT")
print("=" * 70)

# Compute metrics for all communities, sorted by completeness
for comm in sorted(communities.values(), key=lambda c: -c.completeness)[:10]:
    members = [identities[n] for n in comm.members if n in identities]
    comm.tnf_coherence = tnf_coherence(members)
    comm.coverage_correlation = coverage_coherence(members)
    comm.graph_connectivity = graph_connectivity(comm.members, adjacency)

    metrics = community_metrics(comm, identities, adjacency)

    print(f"\n{comm.name} [{comm.quality_tier}]")
    print(f"  Members: {len(comm.members)}, Size: {comm.total_size:,} bp")
    print(f"  Wholeness: {comm.completeness:.0f}%, Redundancy: {comm.redundancy:.0f}%")
    print(f"  TNF coherence:     {comm.tnf_coherence:.4f}")
    print(f"  Coverage coherence: {comm.coverage_correlation:.4f}")
    print(f"  Graph connectivity: {comm.graph_connectivity:.4f}")
    print(f"  Mean fit: {metrics['mean_fit']:+.3f}")
    print(f"  Min fit:  {metrics['min_fit']:+.3f}")
    print(f"  Uneasy members: {metrics['n_uneasy']}")

    # Show any uneasy members
    if metrics['n_uneasy'] > 0:
        for name in comm.members:
            c = identities.get(name)
            if c:
                v = contig_fit_score(c, comm, adjacency)
                if v < 0:
                    print(f"    {name}: fit_score {v:+.3f}, GC={c.gc:.3f} "
                          f"(community mean {comm.mean_gc:.3f}), "
                          f"n_binners={c.n_binners}/5")

# Check some unbinned contigs
print("\n" + "=" * 70)
print("UNBINNED CONTIGS WITH STRONGEST BINNER SUPPORT")
print("=" * 70)

unbinned_supported = [c for c in identities.values()
                   if c.community is None and c.n_binners >= 4]
unbinned_supported.sort(key=lambda c: -c.size)

for c in unbinned_supported[:10]:
    # Find which community they fit best
    best_comm = None
    best_score = -2.0
    for comm in communities.values():
        v = contig_fit_score(c, comm, adjacency)
        if v > best_score:
            best_score = v
            best_comm = comm

    print(f"\n{c.name} ({c.size:,} bp, GC={c.gc:.3f}, n_binners={c.n_binners}/5)")
    print(f"  Binner assignments: {c.binner_assignments}")
    if best_comm:
        print(f"  Best fit: {best_comm.name} (score {best_score:+.3f})")
