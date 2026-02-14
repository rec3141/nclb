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
housed = sum(1 for c in identities.values() if c.community is not None)
unhoused = len(identities) - housed
voiced = sum(1 for c in identities.values() if c.voice_strength > 0)
voiceless = sum(1 for c in identities.values() if c.voice_strength == 0 and c.community is None)
graph_connected = sum(1 for c in identities.values() if c.connections)

print(f"Housed:            {housed:,} ({100*housed/len(identities):.1f}%)")
print(f"Unhoused:          {unhoused:,} ({100*unhoused/len(identities):.1f}%)")
print(f"  with oracle voice: {voiced - housed:,}")
print(f"  truly voiceless:   {voiceless:,}")
print(f"Graph-connected:   {graph_connected:,}")
print()

# Elder hierarchy
for rank in ["contigsattva", "sage", "full", "apprentice", "none"]:
    n = sum(1 for c in communities.values() if c.elder_rank == rank)
    if n > 0:
        print(f"  {rank:15s}: {n} communities")

print()

# Show top 5 communities
print("Top 5 communities by completeness:")
for comm in sorted(communities.values(), key=lambda c: -c.completeness)[:5]:
    print(f"  {comm.name:30s}  {comm.completeness:5.1f}% complete  "
          f"{comm.redundancy:4.1f}% redundant  {len(comm.members)} contigs  "
          f"[{comm.elder_rank}]")

print()

# Show a sample contig identity
sample = identities.get("contig_1") or list(identities.values())[0]
print(f"Sample identity card: {sample.name}")
print(f"  Size: {sample.size:,} bp")
print(f"  GC:   {sample.gc:.3f}")
print(f"  Coverage: {sample.coverage}")
print(f"  Connections: {len(sample.connections)} graph neighbors")
print(f"  Voice strength: {sample.voice_strength}/5")
print(f"  Testimony: {sample.testimony}")
print(f"  Community: {sample.community or 'unhoused'}")
