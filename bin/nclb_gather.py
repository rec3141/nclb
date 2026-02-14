#!/usr/bin/env python3
"""Gathering 1: Self-Knowledge.

Reads all pipeline outputs, builds identity cards for every contig,
computes community profiles with harmony metrics, and outputs
a gathering.json for the Conversations phase.

Usage:
    nclb_gather.py --results /path/to/results [--output gathering.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from bin/ or as installed package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import numpy as np

from nclb.identity import build_identities, load_gfa_graph
from nclb.valence import contig_valence, community_harmony, tnf_coherence, coverage_coherence
from nclb.graph import graph_connectivity, shared_edge_communities
from nclb.resonance import ResonanceMap


def find_binner_paths(binning_dir: Path) -> dict[str, Path]:
    """Auto-discover binner contig_bins.tsv files."""
    binners = {}
    for name in ["semibin", "metabat", "maxbin", "lorbin", "comebin"]:
        tsv = binning_dir / name / "contig_bins.tsv"
        if tsv.exists():
            binners[name] = tsv
    return binners


def serialize_identity(c) -> dict:
    """Convert a ContigIdentity to a JSON-safe dict."""
    d = {
        "name": c.name,
        "size": c.size,
        "gc": round(c.gc, 4),
        "assembly_coverage": round(c.assembly_coverage, 2),
        "is_circular": c.is_circular,
        "is_repeat": c.is_repeat,
        "multiplicity": c.multiplicity,
        "coverage": [round(float(x), 4) for x in c.coverage],
        "connections": c.connections,
        "testimony": c.testimony,
        "voice_strength": c.voice_strength,
        "community": c.community,
        "membership_type": c.membership_type,
        "valence": round(c.valence, 4),
        "ancestry": c.ancestry,
        "gifts": c.gifts,
        "marker_genes": c.marker_genes,
    }
    # MGE annotations (only include if present)
    if c.mge_type:
        d["mge_type"] = c.mge_type
    if c.is_viral:
        d["is_viral"] = True
        d["virus_score"] = round(c.virus_score, 4)
        d["virus_hallmarks"] = c.virus_hallmarks
        if c.virus_taxonomy:
            d["virus_taxonomy"] = c.virus_taxonomy
    if c.is_plasmid:
        d["is_plasmid"] = True
        d["plasmid_score"] = round(c.plasmid_score, 4)
        d["plasmid_hallmarks"] = c.plasmid_hallmarks
        if c.conjugation_genes:
            d["conjugation_genes"] = c.conjugation_genes
        if c.amr_genes:
            d["amr_genes"] = c.amr_genes
    if c.is_provirus:
        d["is_provirus"] = True
        d["proviral_length"] = c.proviral_length
    if c.checkv_quality:
        d["checkv_quality"] = c.checkv_quality
        d["viral_genes"] = c.viral_genes
        d["host_genes"] = c.host_genes
    return d


def serialize_community(comm, harmony_report: dict) -> dict:
    """Convert a CommunityProfile + harmony report to a JSON-safe dict."""
    return {
        "name": comm.name,
        "source_binner": comm.source_binner,
        "members": comm.members,
        "mean_gc": round(comm.mean_gc, 4),
        "gc_stdev": round(comm.gc_stdev, 4),
        "total_size": comm.total_size,
        "n50": comm.n50,
        "completeness": round(comm.completeness, 2),
        "redundancy": round(comm.redundancy, 2),
        "contamination": round(comm.contamination, 2),
        "checkm2_completeness": round(comm.checkm2_completeness, 2),
        "elder_rank": comm.elder_rank,
        "tnf_coherence": round(comm.tnf_coherence, 4),
        "coverage_correlation": round(comm.coverage_correlation, 4),
        "graph_connectivity": round(comm.graph_connectivity, 4),
        "mean_valence": round(harmony_report["mean_valence"], 4),
        "min_valence": round(harmony_report["min_valence"], 4),
        "collective_harmony": round(harmony_report["collective_harmony"], 4),
        "n_uneasy": harmony_report["n_uneasy"],
        "missing_markers": comm.missing_markers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NCLB Gathering 1: Build contig identity cards and community profiles"
    )
    parser.add_argument(
        "--results", "-r", type=Path, required=True,
        help="Path to Nextflow results directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON path (default: <results>/binning/nclb/gathering.json)"
    )
    parser.add_argument(
        "--resonance-k", type=int, default=20,
        help="K for resonance KNN (default: 20)"
    )
    parser.add_argument(
        "--resonance-min-score", type=float, default=0.25,
        help="Minimum resonance score for candidates (default: 0.25)"
    )
    parser.add_argument(
        "--resonance-max-candidates", type=int, default=50,
        help="Max resonance candidates per community (default: 50)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    results = args.results
    output_path = args.output or (results / "binning" / "nclb" / "gathering.json")

    def log(msg: str):
        if not args.quiet:
            print(msg, file=sys.stderr)

    t0 = time.time()

    # --- Verify inputs exist ---
    assembly_dir = results / "assembly"
    mapping_dir = results / "mapping"
    binning_dir = results / "binning"

    required_files = {
        "TNF": assembly_dir / "tnf.tsv",
        "Depths": mapping_dir / "depths.txt",
        "Assembly info": assembly_dir / "assembly_info.txt",
        "Assembly graph": assembly_dir / "assembly_graph.gfa",
        "DAS Tool consensus": binning_dir / "dastool" / "contig2bin.tsv",
        "DAS Tool summary": binning_dir / "dastool" / "summary.tsv",
    }

    missing = {k: v for k, v in required_files.items() if not v.exists()}
    if missing:
        for name, path in missing.items():
            print(f"[ERROR] Missing {name}: {path}", file=sys.stderr)
        sys.exit(1)

    # Optional CheckM2
    checkm2_path = binning_dir / "checkm2" / "quality_report.tsv"
    if not checkm2_path.exists():
        log("[WARNING] No CheckM2 results found — community quality will use DAS Tool SCG metrics only")
        checkm2_path = None

    # Optional MGE data (geNomad + CheckV)
    mge_dir = results / "mge"
    virus_summary_path = mge_dir / "genomad" / "virus_summary.tsv"
    plasmid_summary_path = mge_dir / "genomad" / "plasmid_summary.tsv"
    checkv_quality_path = mge_dir / "checkv" / "quality_summary.tsv"

    if virus_summary_path.exists():
        log(f"[INFO] Found geNomad virus data: {virus_summary_path}")
    else:
        log("[INFO] No geNomad virus data found")
        virus_summary_path = None

    if plasmid_summary_path.exists():
        log(f"[INFO] Found geNomad plasmid data: {plasmid_summary_path}")
    else:
        log("[INFO] No geNomad plasmid data found")
        plasmid_summary_path = None

    if checkv_quality_path.exists():
        log(f"[INFO] Found CheckV quality data: {checkv_quality_path}")
    else:
        log("[INFO] No CheckV quality data found")
        checkv_quality_path = None

    # Auto-discover binners
    binner_paths = find_binner_paths(binning_dir)
    if not binner_paths:
        print("[ERROR] No binner outputs found in {binning_dir}", file=sys.stderr)
        sys.exit(1)
    log(f"[INFO] Found {len(binner_paths)} binners: {', '.join(sorted(binner_paths))}")

    # --- Gathering 1: Build identity cards ---
    log("[INFO] Building identity cards...")
    identities, communities = build_identities(
        tnf_path=required_files["TNF"],
        depths_path=required_files["Depths"],
        assembly_info_path=required_files["Assembly info"],
        gfa_path=required_files["Assembly graph"],
        binner_paths=binner_paths,
        consensus_path=required_files["DAS Tool consensus"],
        summary_path=required_files["DAS Tool summary"],
        checkm2_path=checkm2_path,
        virus_summary_path=virus_summary_path,
        plasmid_summary_path=plasmid_summary_path,
        checkv_quality_path=checkv_quality_path,
    )
    log(f"[INFO] {len(identities):,} contigs, {len(communities)} communities")

    # --- Load assembly graph ---
    log("[INFO] Parsing assembly graph...")
    adjacency = load_gfa_graph(required_files["Assembly graph"])
    log(f"[INFO] {len(adjacency):,} contigs with graph connections")

    # --- Compute community harmony metrics ---
    log("[INFO] Computing community harmony...")
    harmony_reports = {}
    for comm_name, comm in communities.items():
        members = [identities[n] for n in comm.members if n in identities]

        comm.tnf_coherence = tnf_coherence(members)
        comm.coverage_correlation = coverage_coherence(members)
        comm.graph_connectivity = graph_connectivity(comm.members, adjacency)

        harmony = community_harmony(comm, identities, adjacency)
        harmony_reports[comm_name] = harmony

    # --- Compute per-contig valence ---
    log("[INFO] Computing contig valence scores...")
    for name, contig in identities.items():
        if contig.community and contig.community in communities:
            contig.valence = contig_valence(contig, communities[contig.community], adjacency)
        else:
            contig.valence = -1.0  # unhoused baseline

    # --- Build resonance map ---
    log("[INFO] Building resonance map (KNN)...")
    resonance_map = ResonanceMap(
        identities, communities, adjacency, k=args.resonance_k
    )

    # Find resonance candidates for each community
    log("[INFO] Finding resonance candidates for each community...")
    resonance_candidates = {}
    for comm_name, comm in communities.items():
        candidates = resonance_map.find_resonant_contigs(
            comm,
            max_candidates=args.resonance_max_candidates,
            min_score=args.resonance_min_score,
        )
        if candidates:
            resonance_candidates[comm_name] = [
                {
                    "contig": c.contig_name,
                    "tnf_similarity": round(c.tnf_similarity, 4),
                    "coverage_correlation": round(c.coverage_correlation, 4),
                    "graph_edges": c.graph_edges,
                    "voice_agreement": c.voice_agreement,
                    "score": round(c.score, 4),
                }
                for c in candidates
            ]

    # --- Find inter-community graph bridges ---
    log("[INFO] Finding inter-community graph connections...")
    cross_edges = shared_edge_communities(communities, adjacency)

    # --- Assembly statistics ---
    housed = sum(1 for c in identities.values() if c.community is not None)
    unhoused = len(identities) - housed
    voiced_unhoused = sum(
        1 for c in identities.values()
        if c.community is None and c.voice_strength > 0
    )
    graph_connected = sum(1 for c in identities.values() if c.connections)
    voiceless = sum(
        1 for c in identities.values()
        if c.community is None and c.voice_strength == 0 and not c.connections
    )

    assembly_stats = {
        "total_contigs": len(identities),
        "total_communities": len(communities),
        "housed": housed,
        "unhoused": unhoused,
        "voiced_unhoused": voiced_unhoused,
        "graph_connected": graph_connected,
        "truly_voiceless": voiceless,
        "total_assembly_size": sum(c.size for c in identities.values()),
        "housed_size": sum(c.size for c in identities.values() if c.community),
        "unhoused_size": sum(c.size for c in identities.values() if not c.community),
    }

    elder_counts = {}
    for comm in communities.values():
        elder_counts[comm.elder_rank] = elder_counts.get(comm.elder_rank, 0) + 1
    assembly_stats["elder_hierarchy"] = elder_counts

    # MGE stats
    n_viral = sum(1 for c in identities.values() if c.is_viral)
    n_plasmid = sum(1 for c in identities.values() if c.is_plasmid)
    n_provirus = sum(1 for c in identities.values() if c.is_provirus)
    n_viral_housed = sum(1 for c in identities.values() if c.is_viral and c.community)
    n_plasmid_housed = sum(1 for c in identities.values() if c.is_plasmid and c.community)
    n_provirus_housed = sum(1 for c in identities.values() if c.is_provirus and c.community)
    assembly_stats["mge"] = {
        "viral": n_viral,
        "plasmid": n_plasmid,
        "provirus": n_provirus,
        "viral_housed": n_viral_housed,
        "plasmid_housed": n_plasmid_housed,
        "provirus_housed": n_provirus_housed,
    }

    # --- Assemble output ---
    log("[INFO] Serializing gathering data...")
    gathering = {
        "version": "0.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "assembly_stats": assembly_stats,
        "communities": {
            name: serialize_community(comm, harmony_reports[name])
            for name, comm in communities.items()
        },
        "resonance_candidates": resonance_candidates,
        "cross_community_edges": [
            {"community_a": a, "community_b": b, "n_edges": n}
            for a, b, n in cross_edges[:50]  # top 50 pairs
        ],
        "uneasy_members": {},
        "unhoused_with_voice": [],
    }

    # Detailed uneasy member lists per community
    for comm_name, harmony in harmony_reports.items():
        if harmony["n_uneasy"] > 0:
            uneasy = []
            comm = communities[comm_name]
            for name in comm.members:
                c = identities.get(name)
                if c:
                    v = contig_valence(c, comm, adjacency)
                    if v < 0:
                        # Graph neighbors in vs out of community
                        neighbors_in = [n for n in c.connections if n in comm.members]
                        neighbors_out = [n for n in c.connections if n not in comm.members]
                        entry = {
                            "contig": name,
                            "valence": round(v, 4),
                            "size": c.size,
                            "gc": round(c.gc, 4),
                            "community_mean_gc": round(comm.mean_gc, 4),
                            "voice_strength": c.voice_strength,
                            "coverage": [round(float(x), 4) for x in c.coverage],
                            "graph_neighbors_in_community": len(neighbors_in),
                            "graph_neighbors_outside": len(neighbors_out),
                            "total_graph_neighbors": len(c.connections),
                            "testimony": c.testimony,
                        }
                        if c.mge_type:
                            entry["mge_type"] = c.mge_type
                        if c.is_viral:
                            entry["virus_score"] = round(c.virus_score, 4)
                            entry["virus_hallmarks"] = c.virus_hallmarks
                        if c.is_plasmid:
                            entry["plasmid_score"] = round(c.plasmid_score, 4)
                        if c.is_provirus:
                            entry["is_provirus"] = True
                            entry["proviral_length"] = c.proviral_length
                        uneasy.append(entry)
            if uneasy:
                gathering["uneasy_members"][comm_name] = uneasy

    # Top unhoused contigs with voice (for Round 2)
    unhoused_voiced = [
        c for c in identities.values()
        if c.community is None and c.voice_strength >= 2
    ]
    unhoused_voiced.sort(key=lambda c: (-c.voice_strength, -c.size))
    for c in unhoused_voiced[:500]:
        gathering["unhoused_with_voice"].append(serialize_identity(c))

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(gathering, f, indent=2)

    elapsed = time.time() - t0

    # --- Print summary report ---
    log("")
    log("=" * 70)
    log("GATHERING 1: SELF-KNOWLEDGE — COMPLETE")
    log("=" * 70)
    log(f"")
    log(f"  Contigs:       {assembly_stats['total_contigs']:>8,}")
    log(f"  Communities:   {assembly_stats['total_communities']:>8}")
    log(f"  Housed:        {housed:>8,} ({100*housed/len(identities):.1f}%)")
    log(f"  Unhoused:      {unhoused:>8,} ({100*unhoused/len(identities):.1f}%)")
    log(f"    with voice:  {voiced_unhoused:>8,}")
    log(f"    voiceless:   {voiceless:>8,}")
    log(f"    graph-linked:{graph_connected:>8,}")
    log(f"")
    log(f"  Elder hierarchy:")
    for rank in ["contigsattva", "sage", "full", "apprentice", "none"]:
        n = elder_counts.get(rank, 0)
        if n > 0:
            log(f"    {rank:15s}: {n}")
    log(f"")

    mge = assembly_stats["mge"]
    if mge["viral"] or mge["plasmid"] or mge["provirus"]:
        log(f"  Mobile genetic elements:")
        if mge["viral"]:
            log(f"    Viral:     {mge['viral']:>6} ({mge['viral_housed']} housed)")
        if mge["plasmid"]:
            log(f"    Plasmid:   {mge['plasmid']:>6} ({mge['plasmid_housed']} housed)")
        if mge["provirus"]:
            log(f"    Provirus:  {mge['provirus']:>6} ({mge['provirus_housed']} housed)")
        log(f"")

    total_uneasy = sum(h["n_uneasy"] for h in harmony_reports.values())
    log(f"  Total uneasy members: {total_uneasy}")
    log(f"  Resonance candidates: {sum(len(v) for v in resonance_candidates.values())}")
    log(f"  Cross-community edges: {len(cross_edges)}")
    log(f"")
    log(f"  Output: {output_path}")
    log(f"  Time:   {elapsed:.1f}s")
    log("=" * 70)


if __name__ == "__main__":
    main()
