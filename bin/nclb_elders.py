#!/usr/bin/env python3
"""Elder Investigations: SCG Redundancy Analysis.

For each bin with sufficient completeness and redundant single-copy
markers, Elders investigate whether the redundancy represents contamination
or ecotype variation (population-level diversity).

Investigation workflow per redundant marker:
  1. Identify contigs carrying duplicate copies
  2. Compare their taxonomy (Kaiju ancestry) — same species?
  3. Compare their composition (TNF cosine, GC) — how divergent?
  4. Compute ANI between variant-carrying contigs and bin backbone
  5. Check if variants are on MGE contigs (proviruses, plasmids, islands)
  6. Render verdict: ecotype | contamination | mge | uncertain

Usage:
    nclb_elders.py --results /path/to/results [--output elder_reports.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from nclb.identity import (
    build_identities, load_gfa_graph, ContigIdentity, CommunityProfile,
)
from nclb.elders import (
    ElderConfig, ElderToolkit, ElderInvestigation,
    identify_redundant_markers,
)
from nclb.valence import contig_fit_score


# ---------------------------------------------------------------------------
# Investigation logic (deterministic — no LLM needed for core analysis)
# ---------------------------------------------------------------------------

def investigate_redundancy(
    community: CommunityProfile,
    marker_name: str,
    carrier_contigs: list[str],
    identities: dict[str, ContigIdentity],
    adjacency: dict[str, list[str]],
    toolkit: ElderToolkit | None = None,
    assembly_seqs: dict[str, str] | None = None,
) -> dict:
    """Investigate a single redundant marker in a bin.

    Returns a report dict with evidence and verdict.
    """
    carriers = [identities[c] for c in carrier_contigs if c in identities]
    if len(carriers) < 2:
        return {"marker": marker_name, "verdict": "skip", "reason": "fewer than 2 carriers found"}

    report = {
        "marker": marker_name,
        "n_copies": len(carriers),
        "carrier_contigs": [c.name for c in carriers],
        "evidence": {},
    }

    # --- Evidence 1: Taxonomy comparison ---
    ancestries = {}
    for c in carriers:
        if c.ancestry:
            ancestries[c.name] = c.ancestry
    report["evidence"]["taxonomy"] = ancestries

    # Check if all classified carriers share the same lineage
    unique_lineages = set(ancestries.values())
    if len(unique_lineages) == 1 and len(ancestries) == len(carriers):
        report["evidence"]["taxonomy_agreement"] = True
        report["evidence"]["shared_lineage"] = list(unique_lineages)[0]
    elif len(unique_lineages) <= 1:
        report["evidence"]["taxonomy_agreement"] = None  # not enough data
    else:
        report["evidence"]["taxonomy_agreement"] = False

    # --- Evidence 2: Composition comparison ---
    gc_values = [c.gc for c in carriers]
    gc_range = max(gc_values) - min(gc_values)
    report["evidence"]["gc_values"] = {c.name: round(c.gc, 4) for c in carriers}
    report["evidence"]["gc_range"] = round(gc_range, 4)
    report["evidence"]["community_mean_gc"] = round(community.mean_gc, 4)

    # TNF cosine distance between carriers
    if len(carriers) == 2:
        tnf_dist = cosine_distance(carriers[0].tnf, carriers[1].tnf)
        report["evidence"]["tnf_distance_between_carriers"] = round(float(tnf_dist), 4)
    else:
        # Pairwise for multi-copy
        pairwise = []
        for i, a in enumerate(carriers):
            for b in carriers[i + 1:]:
                d = cosine_distance(a.tnf, b.tnf)
                pairwise.append({
                    "a": a.name, "b": b.name,
                    "distance": round(float(d), 4),
                })
        report["evidence"]["tnf_pairwise_distances"] = pairwise

    # Distance from carriers to community centroid
    if community.tnf_centroid is not None:
        centroid_dists = {}
        for c in carriers:
            d = cosine_distance(c.tnf, community.tnf_centroid)
            centroid_dists[c.name] = round(float(d), 4)
        report["evidence"]["tnf_distance_to_centroid"] = centroid_dists

    # --- Evidence 3: Coverage comparison ---
    if community.mean_coverage is not None and len(carriers[0].coverage) > 0:
        cov_data = {}
        for c in carriers:
            cov_data[c.name] = {
                "coverage": [round(float(x), 2) for x in c.coverage],
                "mean": round(float(np.mean(c.coverage)), 2),
            }
        cov_data["community_mean"] = [round(float(x), 2) for x in community.mean_coverage]
        report["evidence"]["coverage"] = cov_data

    # --- Evidence 4: MGE status ---
    mge_carriers = {}
    for c in carriers:
        if c.mge_type:
            mge_carriers[c.name] = {
                "mge_type": c.mge_type,
                "is_viral": c.is_viral,
                "is_plasmid": c.is_plasmid,
                "is_provirus": c.is_provirus,
            }
    report["evidence"]["mge_carriers"] = mge_carriers

    # --- Evidence 5: Graph connectivity ---
    graph_data = {}
    member_set = set(community.members)
    for c in carriers:
        neighbors_in = [n for n in c.connections if n in member_set]
        neighbors_out = [n for n in c.connections if n not in member_set]
        graph_data[c.name] = {
            "graph_neighbors_in_community": len(neighbors_in),
            "graph_neighbors_outside": len(neighbors_out),
            "total_neighbors": len(c.connections),
        }
    report["evidence"]["graph"] = graph_data

    # --- Evidence 6: ANI (if toolkit and sequences available) ---
    if toolkit and assembly_seqs:
        seqs_a = {carriers[0].name: assembly_seqs.get(carriers[0].name, "")}
        seqs_b = {carriers[1].name: assembly_seqs.get(carriers[1].name, "")}
        if seqs_a[carriers[0].name] and seqs_b[carriers[1].name]:
            ani_result = toolkit.compute_ani(
                seqs_a, seqs_b,
                name_a=carriers[0].name, name_b=carriers[1].name,
            )
            report["evidence"]["ani"] = {
                "ani": ani_result.ani,
                "n_aligned": ani_result.n_aligned,
                "coverage": ani_result.coverage,
            }

    # --- Render verdict ---
    report["verdict"] = render_verdict(report)

    return report


def render_verdict(report: dict) -> dict:
    """Render a verdict based on accumulated evidence.

    Returns: {"classification": str, "confidence": float, "reasoning": str}
    """
    evidence = report["evidence"]
    reasons = []
    score_ecotype = 0.0  # positive = ecotype, negative = contamination

    # MGE check — if a carrier is a known MGE, it's not contamination
    mge = evidence.get("mge_carriers", {})
    if mge:
        mge_names = list(mge.keys())
        mge_types = [m["mge_type"] for m in mge.values()]
        reasons.append(f"Carrier(s) {', '.join(mge_names)} are MGE ({', '.join(mge_types)})")
        return {
            "classification": "mge",
            "confidence": 0.9,
            "reasoning": "; ".join(reasons) + ". Redundancy is from a mobile genetic element, not contamination.",
            "reclassify_as": "movable",
            "mge_contigs": mge_names,
        }

    # Taxonomy agreement
    tax_agree = evidence.get("taxonomy_agreement")
    if tax_agree is True:
        score_ecotype += 0.3
        reasons.append(f"All carriers share lineage: {evidence.get('shared_lineage', '?')}")
    elif tax_agree is False:
        score_ecotype -= 0.4
        reasons.append("Carriers have different taxonomic lineages")

    # GC range
    gc_range = evidence.get("gc_range", 0)
    if gc_range < 0.02:
        score_ecotype += 0.2
        reasons.append(f"GC range between carriers is small ({gc_range:.3f})")
    elif gc_range > 0.05:
        score_ecotype -= 0.2
        reasons.append(f"GC range between carriers is large ({gc_range:.3f})")

    # TNF distance between carriers
    tnf_dist = evidence.get("tnf_distance_between_carriers")
    if tnf_dist is not None:
        if tnf_dist < 0.05:
            score_ecotype += 0.2
            reasons.append(f"TNF distance between carriers is small ({tnf_dist:.3f})")
        elif tnf_dist > 0.15:
            score_ecotype -= 0.3
            reasons.append(f"TNF distance between carriers is large ({tnf_dist:.3f})")

    # ANI
    ani_data = evidence.get("ani", {})
    if ani_data:
        ani = ani_data.get("ani", 0)
        if ani >= 99:
            score_ecotype += 0.3
            reasons.append(f"ANI between carriers is {ani:.1f}% (same strain)")
        elif ani >= 95:
            score_ecotype += 0.15
            reasons.append(f"ANI between carriers is {ani:.1f}% (same species)")
        elif ani > 0:
            score_ecotype -= 0.3
            reasons.append(f"ANI between carriers is {ani:.1f}% (different species)")

    # Graph connectivity
    graph = evidence.get("graph", {})
    all_connected = all(
        g.get("graph_neighbors_in_community", 0) > 0
        for g in graph.values()
    )
    if all_connected:
        score_ecotype += 0.1
        reasons.append("All carriers are graph-connected to bin backbone")

    # Final classification
    if score_ecotype >= 0.3:
        classification = "ecotype"
        confidence = min(0.5 + score_ecotype, 0.95)
        reasoning = "; ".join(reasons) + ". This appears to be population-level ecotype variation, not contamination."
    elif score_ecotype <= -0.3:
        classification = "contamination"
        confidence = min(0.5 - score_ecotype, 0.95)
        reasoning = "; ".join(reasons) + ". This appears to be genuine contamination from a different organism."
    else:
        classification = "uncertain"
        confidence = 0.3
        reasoning = "; ".join(reasons) + ". Evidence is ambiguous — manual review recommended."

    return {
        "classification": classification,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NCLB Elder Investigations: analyze SCG redundancy per bin"
    )
    parser.add_argument(
        "--results", "-r", type=Path, required=True,
        help="Path to Nextflow results directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path (default: <results>/binning/nclb/elder_reports.json)"
    )
    parser.add_argument(
        "--min-rank", type=str, default="good",
        choices=["fair", "good", "high", "excellent"],
        help="Minimum quality tier to investigate (default: good)"
    )
    parser.add_argument(
        "--with-ani", action="store_true",
        help="Compute ANI between variant carriers (requires minimap2)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    results = args.results
    output_path = args.output or (results / "binning" / "nclb" / "elder_reports.json")

    def log(msg: str):
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    t0 = time.time()

    # Rank hierarchy for filtering
    rank_order = {"low": 0, "fair": 1, "good": 2, "high": 3, "excellent": 4}
    min_rank_val = rank_order.get(args.min_rank, 2)

    # --- Load identity data ---
    log("[INFO] Loading identity data...")
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

    mge_dir = results / "mge"
    virus_path = mge_dir / "genomad" / "virus_summary.tsv"
    plasmid_path = mge_dir / "genomad" / "plasmid_summary.tsv"
    checkv_path = mge_dir / "checkv" / "quality_summary.tsv"
    kaiju_path = results / "taxonomy" / "kaiju" / "kaiju_contigs.tsv"

    integron_path = mge_dir / "integrons" / "integrons.tsv"
    island_path = mge_dir / "genomic_islands" / "genomic_islands.tsv"
    msf_path = mge_dir / "macsyfinder" / "all_systems.tsv"
    prokka_gff_path = None
    annotation_dir = results / "annotation" / "prokka"
    if annotation_dir.exists():
        gffs = sorted(annotation_dir.glob("*/*.gff")) + sorted(annotation_dir.glob("*.gff"))
        if gffs:
            prokka_gff_path = gffs[-1]

    identities, communities = build_identities(
        tnf_path=assembly_dir / "tnf.tsv",
        depths_path=mapping_dir / "depths.txt",
        assembly_info_path=assembly_dir / "assembly_info.txt",
        gfa_path=assembly_dir / "assembly_graph.gfa",
        binner_paths=binner_paths,
        consensus_path=binning_dir / "dastool" / "contig2bin.tsv",
        summary_path=binning_dir / "dastool" / "summary.tsv",
        checkm2_path=checkm2_path,
        virus_summary_path=virus_path if virus_path.exists() else None,
        plasmid_summary_path=plasmid_path if plasmid_path.exists() else None,
        checkv_quality_path=checkv_path if checkv_path.exists() else None,
        kaiju_taxonomy_path=kaiju_path if kaiju_path.exists() else None,
        integron_path=integron_path if integron_path.exists() else None,
        genomic_island_path=island_path if island_path.exists() else None,
        macsyfinder_path=msf_path if msf_path.exists() else None,
        prokka_gff_path=prokka_gff_path,
    )
    adjacency = load_gfa_graph(assembly_dir / "assembly_graph.gfa")
    log(f"[INFO] Loaded {len(identities):,} contigs, {len(communities)} bins")

    # --- Load assembly sequences if ANI requested ---
    assembly_seqs = None
    toolkit = None
    if args.with_ani:
        assembly_fasta = assembly_dir / "assembly.fasta"
        if assembly_fasta.exists():
            log("[INFO] Loading assembly sequences for ANI computation...")
            assembly_seqs = {}
            current_name = None
            current_seq = []
            with open(assembly_fasta) as f:
                for line in f:
                    if line.startswith(">"):
                        if current_name:
                            assembly_seqs[current_name] = "".join(current_seq)
                        current_name = line.strip()[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())
            if current_name:
                assembly_seqs[current_name] = "".join(current_seq)
            log(f"[INFO] Loaded {len(assembly_seqs):,} sequences")

            work_dir = output_path.parent / "elder_work"
            toolkit = ElderToolkit(ElderConfig(work_dir=work_dir))

    # --- Investigate communities ---
    log("")
    log("=" * 70)
    log("ELDER INVESTIGATIONS: SCG REDUNDANCY ANALYSIS")
    log("=" * 70)

    eligible = {
        name: comm for name, comm in communities.items()
        if rank_order.get(comm.quality_tier, 0) >= min_rank_val
        and comm.redundancy > 0
    }
    log(f"  Bins with rank >= {args.min_rank} and redundancy > 0: {len(eligible)}")

    all_reports = []
    total_redundant = 0
    total_ecotype = 0
    total_contamination = 0
    total_mge = 0
    total_uncertain = 0

    for comm_name in sorted(eligible.keys()):
        comm = eligible[comm_name]
        redundant_markers = identify_redundant_markers(comm, identities)

        if not redundant_markers:
            log(f"  {comm_name} [{comm.quality_tier}]: redundancy={comm.redundancy:.1f}% but no redundant markers found in identity data")
            continue

        log(f"  {comm_name} [{comm.quality_tier}]: {len(redundant_markers)} redundant marker(s)")

        community_report = {
            "bin": comm_name,
            "quality_tier": comm.quality_tier,
            "completeness": round(comm.completeness, 1),
            "redundancy": round(comm.redundancy, 1),
            "n_members": len(comm.members),
            "total_size": comm.total_size,
            "investigations": [],
        }

        for marker, carrier_contigs in redundant_markers:
            total_redundant += 1
            investigation = investigate_redundancy(
                community=comm,
                marker_name=marker,
                carrier_contigs=carrier_contigs,
                identities=identities,
                adjacency=adjacency,
                toolkit=toolkit,
                assembly_seqs=assembly_seqs,
            )
            community_report["investigations"].append(investigation)

            verdict = investigation.get("verdict", {})
            classification = verdict.get("classification", "unknown")
            if classification == "ecotype":
                total_ecotype += 1
            elif classification == "contamination":
                total_contamination += 1
            elif classification == "mge":
                total_mge += 1
            elif classification == "uncertain":
                total_uncertain += 1

            log(f"    {marker}: {len(carrier_contigs)} copies -> {classification} ({verdict.get('confidence', 0):.0%})")

        all_reports.append(community_report)

    # --- Write output ---
    output = {
        "version": "0.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "min_rank": args.min_rank,
        "with_ani": args.with_ani,
        "summary": {
            "communities_investigated": len(all_reports),
            "redundant_markers_investigated": total_redundant,
            "ecotype_verdicts": total_ecotype,
            "contamination_verdicts": total_contamination,
            "mge_verdicts": total_mge,
            "uncertain_verdicts": total_uncertain,
        },
        "reports": all_reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0

    # --- Summary ---
    log("")
    log("=" * 70)
    log("ELDER INVESTIGATIONS: COMPLETE")
    log("=" * 70)
    log(f"")
    log(f"  Bins investigated:  {len(all_reports)}")
    log(f"  Redundant markers analyzed: {total_redundant}")
    log(f"")
    log(f"  Verdicts:")
    log(f"    Ecotype variation:  {total_ecotype}")
    log(f"    Contamination:      {total_contamination}")
    log(f"    Mobile element:     {total_mge}")
    log(f"    Uncertain:          {total_uncertain}")
    log(f"")
    log(f"  Output: {output_path}")
    log(f"  Time:   {elapsed:.1f}s")
    log("=" * 70)


if __name__ == "__main__":
    main()
