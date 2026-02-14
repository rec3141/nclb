#!/usr/bin/env python3
"""Gathering 3: Integration and Harmony Maximization.

Reads proposals.json from Gathering 2, resolves conflicts via the
Mediator, applies changes, extracts FASTAs, and generates the Chronicle.

Usage:
    nclb_integrate.py --proposals proposals.json --results /path/to/results \
                      [--output /path/to/nclb_output]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import numpy as np

from nclb.identity import build_identities, load_gfa_graph, ContigIdentity, CommunityProfile
from nclb.valence import contig_valence, community_harmony, tnf_coherence, coverage_coherence
from nclb.graph import graph_connectivity
from nclb.mediator import extract_proposals, resolve_deterministic, apply_proposals
from nclb.journal import (
    build_chronicle, write_chronicle, write_contig_membership, write_valence_report,
)


def extract_community_fastas(
    communities: dict[str, CommunityProfile],
    identities: dict[str, ContigIdentity],
    assembly_fasta: Path,
    output_dir: Path,
):
    """Extract FASTA files for each community from the co-assembly."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all sequences from the assembly
    sequences: dict[str, str] = {}
    current_name = None
    current_seq = []

    with open(assembly_fasta) as f:
        for line in f:
            if line.startswith(">"):
                if current_name:
                    sequences[current_name] = "".join(current_seq)
                current_name = line.strip()[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
    if current_name:
        sequences[current_name] = "".join(current_seq)

    # Write per-community FASTAs
    for comm_name, comm in communities.items():
        fasta_path = output_dir / f"{comm_name}.fa"
        with open(fasta_path, "w") as f:
            for member in comm.members:
                if member in sequences:
                    f.write(f">{member}\n")
                    seq = sequences[member]
                    # Write in 80-char lines
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")


def recompute_harmony(
    communities: dict[str, CommunityProfile],
    identities: dict[str, ContigIdentity],
    adjacency: dict[str, list[str]],
):
    """Recompute all harmony metrics after proposals are applied."""
    for comm_name, comm in communities.items():
        members = [identities[n] for n in comm.members if n in identities]
        if not members:
            continue

        comm.tnf_coherence = tnf_coherence(members)
        comm.coverage_correlation = coverage_coherence(members)
        comm.graph_connectivity = graph_connectivity(comm.members, adjacency)
        comm.total_size = sum(c.size for c in members)
        comm.mean_gc = float(np.mean([c.gc for c in members]))

        # Recompute centroid
        comm.tnf_centroid = np.mean([c.tnf for c in members], axis=0)
        comm.mean_coverage = np.mean([c.coverage for c in members], axis=0)

    # Recompute all contig valences
    for name, c in identities.items():
        if c.community and c.community in communities:
            c.valence = contig_valence(c, communities[c.community], adjacency)
        else:
            c.valence = -1.0


def main():
    parser = argparse.ArgumentParser(
        description="NCLB Gathering 3: Integration and chronicle generation"
    )
    parser.add_argument(
        "--proposals", "-p", type=Path, required=True,
        help="Path to proposals.json from Gathering 2"
    )
    parser.add_argument(
        "--results", "-r", type=Path, required=True,
        help="Path to Nextflow results directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: <results>/binning/nclb)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    results = args.results
    output_dir = args.output or (results / "binning" / "nclb")

    def log(msg: str):
        if not args.quiet:
            print(msg, file=sys.stderr)

    t0 = time.time()

    # --- Load proposals ---
    log("[INFO] Loading proposals...")
    with open(args.proposals) as f:
        proposals_data = json.load(f)

    # --- Rebuild identity data ---
    log("[INFO] Rebuilding identity data...")
    assembly_dir = results / "assembly"
    mapping_dir = results / "mapping"
    binning_dir = results / "binning"

    binner_paths = {}
    for name in ["semibin", "metabat", "maxbin", "lorbin", "comebin"]:
        tsv = binning_dir / name / "contig_bins.tsv"
        if tsv.exists():
            binner_paths[name] = tsv

    checkm2_path = binning_dir / "checkm2" / "quality_report.tsv"
    if not checkm2_path.exists():
        checkm2_path = None

    identities, communities = build_identities(
        tnf_path=assembly_dir / "tnf.tsv",
        depths_path=mapping_dir / "depths.txt",
        assembly_info_path=assembly_dir / "assembly_info.txt",
        gfa_path=assembly_dir / "assembly_graph.gfa",
        binner_paths=binner_paths,
        consensus_path=binning_dir / "dastool" / "contig2bin.tsv",
        summary_path=binning_dir / "dastool" / "summary.tsv",
        checkm2_path=checkm2_path,
    )
    adjacency = load_gfa_graph(assembly_dir / "assembly_graph.gfa")

    # Save before stats
    housed_before = sum(1 for c in identities.values() if c.community)

    # --- Extract and resolve proposals ---
    log("[INFO] Extracting proposals from conversation results...")
    raw_proposals = extract_proposals(proposals_data.get("proposals", []))

    log(f"  Releases: {len(raw_proposals['releases'])}")
    log(f"  Joins: {len(raw_proposals['joins'])}")
    log(f"  Waits: {len(raw_proposals['waits'])}")
    log(f"  Wanders: {len(raw_proposals['wanders'])}")
    log(f"  New communities: {len(raw_proposals['new_communities'])}")
    log(f"  Recruits: {len(raw_proposals['recruits'])}")

    log("[INFO] Resolving conflicts...")
    resolved = resolve_deterministic(raw_proposals, identities, communities, adjacency)

    # --- Apply proposals ---
    log("[INFO] Applying proposals...")
    changes = apply_proposals(resolved, identities, communities)

    log(f"  Released: {len(changes['released'])}")
    log(f"  Joined: {len(changes['joined'])}")
    log(f"  New communities: {len(changes['new_communities_founded'])}")

    # --- Recompute harmony ---
    log("[INFO] Recomputing harmony metrics...")
    recompute_harmony(communities, identities, adjacency)

    # After stats
    housed_after = sum(1 for c in identities.values() if c.community)

    # --- Assembly stats ---
    assembly_stats = {
        "total_contigs": len(identities),
        "total_assembly_size": sum(c.size for c in identities.values()),
        "total_communities": len(communities),
        "housed_before": housed_before,
        "housed_before_pct": 100 * housed_before / len(identities),
        "housed_after": housed_after,
        "housed_after_pct": 100 * housed_after / len(identities),
        "delta_housed": housed_after - housed_before,
        "elder_hierarchy": {},
    }
    for comm in communities.values():
        rank = comm.elder_rank
        assembly_stats["elder_hierarchy"][rank] = assembly_stats["elder_hierarchy"].get(rank, 0) + 1

    # --- Build chronicle ---
    log("[INFO] Building chronicle...")
    chronicle = build_chronicle(assembly_stats, resolved, changes, communities, identities)

    # --- Write outputs ---
    log("[INFO] Writing outputs...")

    # Chronicle
    write_chronicle(chronicle, output_dir)

    # Membership tables
    write_contig_membership(identities, output_dir)

    # Valence report
    write_valence_report(identities, output_dir)

    # Community FASTAs
    assembly_fasta = assembly_dir / "assembly.fasta"
    if assembly_fasta.exists():
        extract_community_fastas(communities, identities, assembly_fasta, output_dir / "communities")
    else:
        log("[WARNING] Assembly FASTA not found — skipping community FASTA extraction")

    # Quality summary
    quality_path = output_dir / "quality_report.tsv"
    with open(quality_path, "w") as f:
        f.write("community\tn_members\ttotal_size\tcompleteness\tredundancy\t"
                "tnf_coherence\tcov_correlation\tgraph_connectivity\t"
                "mean_valence\tmin_valence\telder_rank\n")
        for name, comm in sorted(communities.items()):
            harmony = community_harmony(comm, identities, adjacency)
            f.write(f"{name}\t{len(comm.members)}\t{comm.total_size}\t"
                    f"{comm.completeness:.2f}\t{comm.redundancy:.2f}\t"
                    f"{comm.tnf_coherence:.4f}\t{comm.coverage_correlation:.4f}\t"
                    f"{comm.graph_connectivity:.4f}\t"
                    f"{harmony['mean_valence']:.4f}\t{harmony['min_valence']:.4f}\t"
                    f"{comm.elder_rank}\n")

    elapsed = time.time() - t0

    # --- Summary ---
    log("")
    log("=" * 70)
    log("GATHERING 3: INTEGRATION — COMPLETE")
    log("=" * 70)
    log(f"")
    log(f"  Before NCLB:  {housed_before:,} contigs housed ({100*housed_before/len(identities):.1f}%)")
    log(f"  After NCLB:   {housed_after:,} contigs housed ({100*housed_after/len(identities):.1f}%)")
    log(f"  Delta:        +{housed_after - housed_before:,}")
    log(f"  Communities:  {len(communities)}")
    log(f"")
    log(f"  Released:     {len(changes['released'])}")
    log(f"  Placed:       {len(changes['joined'])}")
    log(f"  New communities: {len(changes['new_communities_founded'])}")
    log(f"")
    log(f"  Output:       {output_dir}")
    log(f"  Chronicle:    {output_dir / 'chronicle.md'}")
    log(f"  Time:         {elapsed:.1f}s")
    log("=" * 70)


if __name__ == "__main__":
    main()
