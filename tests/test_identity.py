#!/usr/bin/env python3
"""Smoke test: load real pipeline data and build identity cards."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from nclb.identity import build_identities

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

print(f"Contigs:     {len(identities):,}")
print(f"Communities: {len(communities)}")
print()

# Population stats
binned = sum(1 for c in identities.values() if c.community is not None)
unbinned = len(identities) - binned
with_binners = sum(1 for c in identities.values() if c.n_binners > 0)
no_binners = sum(1 for c in identities.values() if c.n_binners == 0 and c.community is None)
graph_connected = sum(1 for c in identities.values() if c.connections)

print(f"Binned:            {binned:,} ({100*binned/len(identities):.1f}%)")
print(f"Unbinned:          {unbinned:,} ({100*unbinned/len(identities):.1f}%)")
print(f"  with binner support: {with_binners - binned:,}")
print(f"  no binner support:   {no_binners:,}")
print(f"Graph-connected:   {graph_connected:,}")
print()

# Quality tiers
for tier in ["excellent", "high", "good", "fair", "low"]:
    n = sum(1 for c in communities.values() if c.quality_tier == tier)
    if n > 0:
        print(f"  {tier:15s}: {n} communities")

print()

# Show top 5 communities
print("Top 5 communities by completeness:")
for comm in sorted(communities.values(), key=lambda c: -c.completeness)[:5]:
    print(f"  {comm.name:30s}  {comm.completeness:5.1f}% complete  "
          f"{comm.redundancy:4.1f}% redundant  {len(comm.members)} contigs  "
          f"[{comm.quality_tier}]")

print()

# Show a sample contig identity
sample = identities.get("contig_1") or list(identities.values())[0]
print(f"Sample identity card: {sample.name}")
print(f"  Size: {sample.size:,} bp")
print(f"  GC:   {sample.gc:.3f}")
print(f"  Coverage: {sample.coverage}")
print(f"  Connections: {len(sample.connections)} graph neighbors")
print(f"  Binner count: {sample.n_binners}/5")
print(f"  Binner assignments: {sample.binner_assignments}")
print(f"  Community: {sample.community or 'unbinned'}")
