#!/usr/bin/env python3
"""Gathering 2: The Conversations (tool-use architecture).

Each contig is investigated through tool calls.
The LLM investigates by calling tools like get_contig_info(), compare_to_bin(),
get_bin_info() — each returns actual data, not pre-digested metrics.

Supports OpenAI-compatible servers (LM Studio, ollama, vLLM) and Anthropic API.

Usage:
    nclb_converse.py --results /path/to/results [--output proposals.json]
    nclb_converse.py --results /path/to/results --backend anthropic
    nclb_converse.py --results /path/to/results --base-url http://host:1234/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from nclb.identity import (
    build_identities, load_gfa_graph, load_read_adjacency, merge_adjacencies,
    load_bakta_annotations,
    ContigIdentity, CommunityProfile,
)
from nclb.voices import (
    ContigToolkit, CONTIG_TOOLS_OPENAI, parse_json_response,
)
from nclb.valence import contig_fit_score, tnf_coherence, coverage_coherence


# ---------------------------------------------------------------------------
# System prompts (brief — the LLM discovers context through tools)
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts/ directory. Re-reads on every call so
    prompts can be edited while the pipeline is running."""
    path = _PROMPT_DIR / name
    return path.read_text().strip()


# ---------------------------------------------------------------------------
# Tool-use conversation loop
# ---------------------------------------------------------------------------

def _summarize_tool_result(tool_name: str, result: dict) -> str:
    """Produce a short human-readable summary of a tool result for logging."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    if tool_name == "get_contig_info":
        parts = [f"{result.get('len', 0):,}bp"]
        if result.get("ancestry"):
            # Show last two ranks of lineage
            lineage = result["ancestry"].split(";")
            parts.append(lineage[-1].strip() if lineage else "?")
        if result.get("mge"):
            parts.append(result["mge"])
        if result.get("genes"):
            parts.append(f"{len(result['genes'])} genes")
        if result.get("scgs"):
            parts.append(f"{len(result['scgs'])} SCGs")
        if result.get("defense"):
            parts.append(f"{len(result['defense'])} defense")
        if result.get("integrons"):
            parts.append("integron")
        if result.get("secretion"):
            parts.append("secretion")
        domain = result.get("domain", "unknown")
        if domain != "unknown":
            parts.append(domain)
        n_cds = result.get("n_cds", 0)
        if n_cds:
            parts.append(f"{n_cds} CDS")
        if result.get("rrna"):
            parts.append(f"rRNA:{','.join(result['rrna'])}")
        if result.get("n_ko"):
            parts.append(f"{result['n_ko']} KO")
        if result.get("n_cazymes"):
            parts.append(f"{result['n_cazymes']} CAZy")
        if result.get("n_euk_genes"):
            parts.append(f"{result['n_euk_genes']} euk_genes")
        parts.append(f"n_binners={result.get('n_binners', 0)}")
        return ", ".join(parts)

    if tool_name == "get_binner_assignments":
        binner_assignments = result.get("binner_assignments", {})
        assigned = [f"{k}={v}" for k, v in binner_assignments.items() if v]
        return f"n_binners={result.get('n_binners', 0)}, {', '.join(assigned) or 'none'}"

    if tool_name == "compare_to_bin":
        cov = result.get('cov_r')
        graph = result.get('graph_frac')
        cov_s = f"{cov:.3f}" if cov is not None else "NA"
        graph_s = f"{graph:.3f}" if graph is not None else "NA"
        gc_z = result.get('gc_z', 0)
        tnf_z = result.get('tnf_z', 0)
        return (
            f"fit={result.get('fit', 0):+.3f}, "
            f"tnf_cos={result.get('tnf_cos', 0):.3f}, "
            f"TNF_z={tnf_z:.2f}, "
            f"cov_r={cov_s}, "
            f"graph_frac={graph_s}, "
            f"GC_z={gc_z:.2f}"
        )

    if tool_name == "get_bin_info":
        return (
            f"{result.get('n_members', 0)} members, "
            f"{result.get('len', 0):,}bp, "
            f"{result.get('complete', 0):.0f}% complete, "
            f"{result.get('tier', '?')}, "
            f"{result.get('n_missing_scg', 0)} missing SCGs"
        )

    if tool_name == "get_missing_markers":
        return f"{result.get('n_missing', 0)} missing markers"

    if tool_name == "predict_join_impact":
        return (
            f"+{result.get('len_delta', 0):,}bp, "
            f"GC_shift={result.get('gc_shift', 0):.4f}, "
            f"{result.get('n_contributed', 0)} contributed SCGs"
        )

    if tool_name == "find_graph_connections":
        conns = result.get("bin_connections", [])
        if conns:
            top = ", ".join(f"{c['bin']}({c['edges']}e,{c['reads']}r)" for c in conns[:3])
            return f"{len(conns)} bins: {top}"
        return "no graph connections"

    if tool_name == "get_graph_neighbors":
        neighbors = result.get("neighbors", [])
        read_linked = sum(1 for n in neighbors if n.get("reads", 0) > 0)
        return f"{len(neighbors)} neighbors ({read_linked} read-linked)"

    if tool_name == "read_annotations":
        return (
            f"page {result.get('page', 1)}/{result.get('total_pages', 1)}, "
            f"{result.get('total_features', 0)} total features"
        )

    if tool_name == "get_taxonomy":
        n = result.get("n_sources_classified", 0)
        agree = result.get("genus_agreement", False)
        lineage = result.get("primary_lineage", "?")
        # Show last rank of lineage
        if lineage and ";" in lineage:
            lineage = lineage.split(";")[-1].strip()
        return f"{n} sources, agree={agree}, {lineage}"

    if tool_name == "get_bin_metabolism":
        n = result.get("n_modules", 0)
        n_complete = result.get("n_complete", 0)
        return f"{n} modules ({n_complete} >=75% complete)"

    if tool_name == "render_umap_neighborhood":
        n_nearby = result.get("n_nearby", 0)
        n_bins = len(result.get("nearby_bins", []))
        return f"{n_nearby} contigs, {n_bins} bins [IMAGE]"

    if tool_name == "render_graph_neighborhood":
        n_nodes = result.get("n_nodes", 0)
        return f"{n_nodes} nodes [IMAGE]"

    # Fallback
    return json.dumps(result, default=str)[:120]


def run_tool_conversation(
    client,
    model: str,
    system: str,
    user_prompt: str,
    toolkit: ContigToolkit,
    max_rounds: int = 15,
    log_fn=None,
) -> dict:
    """Run a tool-use conversation loop with an OpenAI-compatible API.

    The LLM calls tools to investigate, we execute and return results,
    until it produces a final text response.

    When approaching the round limit, injects a nudge asking the model to
    produce its final JSON decision, then forces a tool-free final round.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    tool_calls_total = 0
    nudged = False
    recent_call_sigs: list[str] = []  # track per-round call signatures for loop detection
    # Reserve last 3 rounds for winding down: nudge at -3, remind at -2,
    # force text at -1
    wind_down_start = max(max_rounds - 3, max_rounds // 2)

    for round_num in range(max_rounds):
        # Wind-down phase: nudge the model to produce decisions
        if round_num == wind_down_start and not nudged:
            messages.append({
                "role": "user",
                "content": (
                    "You have used most of your investigation budget. "
                    "Please finish your investigation and respond with "
                    "your final JSON decision now. Remember the required "
                    "format from your instructions."
                ),
            })
            nudged = True
        elif round_num == max_rounds - 2 and nudged:
            messages.append({
                "role": "user",
                "content": (
                    "This is your LAST chance to investigate. On the next "
                    "round you must produce your JSON response. Respond "
                    "with the JSON now if you are ready."
                ),
            })

        # On the final round, don't offer tools — force a text response
        is_final_round = (round_num == max_rounds - 1)
        call_kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
        )
        if not is_final_round:
            call_kwargs["tools"] = CONTIG_TOOLS_OPENAI

        try:
            response = client.chat.completions.create(**call_kwargs)
        except Exception as e:
            if log_fn:
                log_fn(f"    [ERROR] API call failed: {e}")
            return {"error": str(e), "tool_calls": tool_calls_total}

        choice = response.choices[0]
        msg = choice.message

        # Check for tool calls
        if msg.tool_calls:
            # Append the assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Log any reasoning text the LLM produced alongside tool calls
            if msg.content and log_fn:
                # Truncate long reasoning to keep logs manageable
                reasoning = msg.content.strip()
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
                log_fn(f"    [REASONING] {reasoning}")

            # Execute each tool call
            for tc in msg.tool_calls:
                tool_calls_total += 1
                try:
                    args = json.loads(tc.function.arguments)
                    result = toolkit.dispatch(tc.function.name, args)
                except Exception as e:
                    result = {"error": f"Tool execution failed: {e}"}

                # Verbose tool logging
                if log_fn:
                    args_short = json.dumps(args, default=str)
                    if len(args_short) > 100:
                        args_short = args_short[:100] + "..."
                    # Summarize result
                    result_summary = _summarize_tool_result(tc.function.name, result)
                    log_fn(f"    [{tool_calls_total}] {tc.function.name}({args_short}) → {result_summary}")

                # Image-aware tool result: send mixed content for VL models
                if "image_base64" in result:
                    img_b64 = result.pop("image_base64")
                    img_fmt = result.pop("image_format", "png")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": [
                            {"type": "text", "text": json.dumps(result, default=str)},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/{img_fmt};base64,{img_b64}",
                            }},
                        ],
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    })

            # Loop detection: if same tools called 3 rounds in a row, break
            round_sig = "|".join(
                f"{tc.function.name}:{tc.function.arguments}"
                for tc in msg.tool_calls
            )
            recent_call_sigs.append(round_sig)
            if len(recent_call_sigs) >= 3 and len(set(recent_call_sigs[-3:])) == 1:
                if log_fn:
                    log_fn("    [WARNING] Loop detected — same tools called 3x, forcing decision")
                messages.append({
                    "role": "user",
                    "content": (
                        "You are repeating the same tool calls. You already have "
                        "all the information you need. Produce your final JSON "
                        "response NOW based on the evidence gathered so far."
                    ),
                })
                nudged = True
                # Jump to wind-down
                wind_down_start = min(wind_down_start, round_num + 1)
            continue

        # No tool calls — this is the final response
        text = msg.content or ""
        if text:
            try:
                parsed = parse_json_response(text)
                parsed["_tool_calls"] = tool_calls_total
                return parsed
            except (json.JSONDecodeError, ValueError):
                # Model produced text but not valid JSON (e.g. XML tool
                # calls when tools were removed). If this is the forced
                # final round, retry once asking for pure JSON.
                if is_final_round:
                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "That was not valid JSON. Please respond with "
                            "ONLY a JSON object matching the required format. "
                            "No tool calls, no XML, no commentary — just the "
                            "JSON object."
                        ),
                    })
                    try:
                        retry = client.chat.completions.create(
                            model=model, messages=messages,
                            max_tokens=4096, temperature=0.1,
                        )
                        retry_text = retry.choices[0].message.content or ""
                        parsed = parse_json_response(retry_text)
                        parsed["_tool_calls"] = tool_calls_total
                        return parsed
                    except Exception:
                        pass
                return {"raw_response": text, "_tool_calls": tool_calls_total}

        # Empty response — shouldn't happen
        if log_fn:
            log_fn(f"    [WARNING] Empty response at round {round_num}")
        break

    # Exhausted rounds — try one final forced-text round without tools
    if log_fn:
        log_fn(f"    [INFO] Forcing final response after {tool_calls_total} tool calls")
    messages.append({
        "role": "user",
        "content": (
            "Investigation complete. You MUST respond now with your final "
            "JSON decision. No more tool calls. Produce ONLY the JSON object "
            "matching the required format from your instructions."
        ),
    })
    try:
        final = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=4096, temperature=0.1,
        )
        text = final.choices[0].message.content or ""
        if text:
            try:
                parsed = parse_json_response(text)
                parsed["_tool_calls"] = tool_calls_total
                return parsed
            except (json.JSONDecodeError, ValueError):
                return {"raw_response": text, "_tool_calls": tool_calls_total}
    except Exception as e:
        if log_fn:
            log_fn(f"    [ERROR] Final forced response failed: {e}")

    return {"error": "max_rounds_exhausted", "_tool_calls": tool_calls_total}


# ---------------------------------------------------------------------------
# Round builders (brief prompts — tools provide the data)
# ---------------------------------------------------------------------------

def round1_prompt(comm_name: str, comm_data: dict,
                  member_scores: list[tuple[str, float]],
                  display_name: str | None = None) -> str:
    """Build Round 1 user prompt from external template."""
    shown_name = display_name or comm_name

    # Show all members sorted by fit score (worst first)
    score_lines = []
    for name, score in sorted(member_scores, key=lambda x: (x[1] is None, x[1] or 0)):
        s = f"{score:+.3f}" if score is not None else "NA"
        score_lines.append(f"  {name}: {s}")

    mean_fit = comm_data.get('mean_fit') or 0.0
    min_fit = comm_data.get('min_fit') or 0.0
    completeness = comm_data.get('completeness') or 0.0
    redundancy = comm_data.get('redundancy') or 0.0

    return _load_prompt("round1_user.txt").format(
        bin_name=shown_name,
        quality_tier=comm_data.get('quality_tier', 'none'),
        n_members=len(comm_data.get('members', [])),
        total_length=f"{comm_data.get('total_size', 0):,}",
        completeness=f"{completeness:.1f}",
        redundancy=f"{redundancy:.1f}",
        mean_fit=f"{mean_fit:+.3f}",
        min_fit=f"{min_fit:+.3f}",
        members_section="\n".join(score_lines),
    )


def round2_prompt(
    contig_names: list[str],
    contig_candidates: dict[str, list[str]] | None = None,
    community_names: dict[str, str] | None = None,
) -> str:
    """Build Round 2 user prompt from external template."""
    display = community_names or {}
    lines = []
    for name in contig_names:
        candidates = contig_candidates.get(name, []) if contig_candidates else []
        if candidates:
            display_names = [display.get(c, c) for c in candidates[:5]]
            lines.append(f"  {name}: candidates → {', '.join(display_names)}")
        else:
            lines.append(f"  {name}: no pre-computed candidates (use find_graph_connections)")

    return _load_prompt("round2_user.txt").format(
        n_contigs=len(contig_names),
        contig_section="\n".join(lines),
    )


def round3_prompt(clusters: list[dict]) -> str:
    """Build Round 3 user prompt from external template."""
    lines = []
    for cl in clusters:
        lines.append(
            f"  Cluster {cl['id']}: {cl['n_members']} contigs, "
            f"{cl['total_size']:,} bp, TNF coherence={cl['tnf_coherence']:.3f}, "
            f"Coverage correlation={cl.get('coverage_correlation', 0):.3f}, "
            f"Mean GC={cl.get('mean_gc', 0):.3f}"
        )
    return _load_prompt("round3_user.txt").format(
        n_clusters=len(clusters),
        cluster_section="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# HDBSCAN clustering for Round 3
# ---------------------------------------------------------------------------

def cluster_voiceless(identities: dict) -> list[dict]:
    """Cluster truly voiceless contigs using HDBSCAN."""
    try:
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return []

    import numpy as np

    voiceless = []
    for c in identities.values():
        if c.community is None and c.n_binners == 0 and not c.connections:
            if c.size >= 5000:
                voiceless.append(c)

    if len(voiceless) < 10:
        return []

    tnf_matrix = np.array([c.tnf for c in voiceless])
    cov_matrix = np.array([c.coverage for c in voiceless])

    scaler_tnf = StandardScaler()
    scaler_cov = StandardScaler()
    tnf_scaled = scaler_tnf.fit_transform(tnf_matrix)
    cov_scaled = scaler_cov.fit_transform(cov_matrix) if cov_matrix.shape[1] > 0 else np.zeros((len(voiceless), 0))

    features = np.hstack([tnf_scaled * 2.0, cov_scaled])
    clusterer = HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(features)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        members = [voiceless[i] for i, l in enumerate(labels) if l == label]
        total_size = sum(c.size for c in members)
        if total_size < 100000:
            continue
        clusters.append({
            "id": int(label),
            "n_members": len(members),
            "total_size": total_size,
            "member_names": [c.name for c in members],
            "tnf_coherence": float(tnf_coherence(members)[0]),
            "coverage_correlation": float(coverage_coherence(members)),
            "mean_gc": float(np.mean([c.gc for c in members])),
        })
    return clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NCLB Gathering 2: Tool-use conversations"
    )
    parser.add_argument(
        "--results", "-r", type=Path, required=True,
        help="Path to Nextflow results directory"
    )
    parser.add_argument(
        "--gathering", "-g", type=Path, default=None,
        help="Path to gathering.json (for pre-computed resonance data)"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output proposals.json path"
    )
    parser.add_argument(
        "--base-url", type=str, default="http://10.151.30.147:1234/v1",
        help="Base URL for OpenAI-compatible server"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="qwen/qwen3-coder-30b",
        help="Model name (default: qwen/qwen3-coder-30b)"
    )
    parser.add_argument(
        "--max-tool-rounds", type=int, default=15,
        help="Max tool-use rounds per conversation (default: 15)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Contigs per Round 2 conversation (default: 5)"
    )
    parser.add_argument(
        "--max-round2", type=int, default=100,
        help="Max Round 2 contigs to process (default: 100)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    results = args.results
    gathering_path = args.gathering or (results / "binning" / "nclb" / "gathering.json")
    output_path = args.output or (results / "binning" / "nclb" / "proposals.json")

    def log(msg: str):
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    t0 = time.time()

    # --- Load full identity data ---
    log("[INFO] Loading identity data from results directory...")
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

    # MGE data
    mge_dir = results / "mge"
    virus_path = mge_dir / "genomad" / "virus_summary.tsv"
    plasmid_path = mge_dir / "genomad" / "plasmid_summary.tsv"
    checkv_path = mge_dir / "checkv" / "quality_summary.tsv"

    # Taxonomy
    kaiju_path = results / "taxonomy" / "kaiju" / "kaiju_contigs.tsv"
    sendsketch_path = results / "taxonomy" / "sendsketch" / "sendsketch_contigs.tsv"
    kraken2_path = results / "taxonomy" / "kraken2" / "kraken2_contigs.tsv"

    # Integrons, genomic islands, secretion systems
    integron_path = mge_dir / "integrons" / "integrons.tsv"
    island_path = mge_dir / "genomic_islands" / "genomic_islands.tsv"
    msf_path = mge_dir / "macsyfinder" / "all_systems.tsv"
    df_path = mge_dir / "defensefinder" / "genes.tsv"
    prokka_gff_path = None
    annotation_dir = results / "annotation" / "prokka"
    if annotation_dir.exists():
        gffs = sorted(annotation_dir.glob("*/*.gff")) + sorted(annotation_dir.glob("*.gff"))
        if gffs:
            prokka_gff_path = gffs[-1]

    # SCG files from DAS Tool
    bacteria_scg = binning_dir / "dastool" / "bacteria.scg"
    archaea_scg = binning_dir / "dastool" / "archaea.scg"

    # Eukaryotic classification (Tiara + Whokaryote consensus)
    eukaryotic_path = results / "eukaryotic" / "consensus" / "contig_classifications.tsv"

    # Bakta full-db annotation (separate from Prokka annotation dir)
    bakta_path = results / "annotation_bakta" / "annotation.tsv"

    # rRNA, metabolism, MetaEuk, KEGG modules
    rrna_path = results / "taxonomy" / "rrna" / "rrna_contigs.tsv"
    metabolism_path = results / "metabolism" / "merged" / "merged_annotations.tsv"
    metaeuk_gff_path = results / "eukaryotic" / "metaeuk" / "metaeuk.gff"
    kegg_modules_path = results / "metabolism" / "modules" / "module_completeness.tsv"

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
        sendsketch_taxonomy_path=sendsketch_path if sendsketch_path.exists() else None,
        kraken2_taxonomy_path=kraken2_path if kraken2_path.exists() else None,
        integron_path=integron_path if integron_path.exists() else None,
        genomic_island_path=island_path if island_path.exists() else None,
        macsyfinder_path=msf_path if msf_path.exists() else None,
        defensefinder_path=df_path if df_path.exists() else None,
        prokka_gff_path=prokka_gff_path,
        bacteria_scg_path=bacteria_scg if bacteria_scg.exists() else None,
        archaea_scg_path=archaea_scg if archaea_scg.exists() else None,
        eukaryotic_path=eukaryotic_path if eukaryotic_path.exists() else None,
        bakta_tsv_path=bakta_path if bakta_path.exists() else None,
        rrna_path=rrna_path if rrna_path.exists() else None,
        metabolism_path=metabolism_path if metabolism_path.exists() else None,
        metaeuk_gff_path=metaeuk_gff_path if metaeuk_gff_path.exists() else None,
        kegg_modules_path=kegg_modules_path if kegg_modules_path.exists() else None,
    )
    gfa_adjacency = load_gfa_graph(assembly_dir / "assembly_graph.gfa")

    # Load read-bridged adjacency from BAMs (supplementary alignments)
    # Cache to JSON to avoid re-scanning 1.8GB of BAMs every startup (~30s)
    bam_files = sorted(mapping_dir.glob("*.sorted.bam"))
    read_adj_cache = mapping_dir / "read_adjacency.json"
    read_adj: dict[str, dict[str, int]] = {}
    if bam_files:
        if read_adj_cache.exists():
            with open(read_adj_cache) as f:
                raw = json.load(f)
            # Handle old format (list) → convert to weighted dict with count 1
            if raw and isinstance(next(iter(raw.values())), list):
                read_adj = {k: {n: 1 for n in v} for k, v in raw.items()}
                log(f"[INFO] Converted legacy read adjacency cache → weighted")
                with open(read_adj_cache, "w") as f:
                    json.dump(read_adj, f)
            else:
                read_adj = raw
            log(f"[INFO] Loaded cached read adjacency ({len(read_adj)} contigs)")
        else:
            read_adj = load_read_adjacency(bam_files)
            with open(read_adj_cache, "w") as f:
                json.dump(read_adj, f)
            log(f"[INFO] Built read adjacency from BAMs ({len(read_adj)} contigs), cached")
        adjacency = merge_adjacencies(gfa_adjacency, read_adj)
        log(f"[INFO] Adjacency: GFA={len(gfa_adjacency)} contigs, "
            f"read-bridged={len(read_adj)} contigs, "
            f"merged={len(adjacency)} contigs")
        # Update contig identity connections with merged adjacency
        for name, c in identities.items():
            c.connections = adjacency.get(name, {})
    else:
        adjacency = {k: {n: 0 for n in v} for k, v in gfa_adjacency.items()}
        for name, c in identities.items():
            c.connections = adjacency.get(name, {})
        log(f"[INFO] No BAM files found in {mapping_dir}, using GFA adjacency only")

    log(f"[INFO] Loaded {len(identities):,} contigs, {len(communities)} DAS Tool communities")

    # Log taxonomy source stats
    tax_counts = {}
    for c in identities.values():
        for src in c.taxonomy:
            tax_counts[src] = tax_counts.get(src, 0) + 1
    if tax_counts:
        parts = [f"{src}={n:,}" for src, n in sorted(tax_counts.items())]
        log(f"[INFO] Taxonomy sources: {', '.join(parts)}")

    # Log eukaryotic classification stats
    n_euk = sum(1 for c in identities.values() if c.domain_class == "eukaryotic")
    n_org = sum(1 for c in identities.values() if c.domain_class == "organellar")
    if n_euk or n_org:
        log(f"[INFO] Domain classification: {n_euk} eukaryotic, {n_org} organellar contigs")

    # Log new data source stats
    n_rrna = sum(1 for c in identities.values() if c.n_rrna_genes > 0)
    n_ko = sum(1 for c in identities.values() if c.n_ko > 0)
    n_euk_genes = sum(1 for c in identities.values() if c.n_euk_genes > 0)
    n_kegg_bins = sum(1 for c in communities.values() if c.kegg_modules)
    if n_rrna or n_ko or n_euk_genes or n_kegg_bins:
        log(f"[INFO] New data: rRNA={n_rrna} contigs, KO={n_ko} contigs, "
            f"MetaEuk={n_euk_genes} contigs, KEGG modules={n_kegg_bins} bins")

    # Seed additional communities from binner agreement
    from nclb.identity import seed_communities_from_binner_agreement
    new_comms, binner_assigned = seed_communities_from_binner_agreement(identities)
    if new_comms:
        for contig, comm_name in binner_assigned.items():
            identities[contig].community = comm_name
            identities[contig].membership_type = "core"
        # Compute marker gene inventory for seeded communities
        from nclb.identity import load_scg_assignments as _load_scg
        _, full_scg_set = _load_scg(
            bacteria_scg if bacteria_scg.exists() else None,
            archaea_scg if archaea_scg.exists() else None,
        )
        for comm in new_comms.values():
            inventory = set()
            for m in comm.members:
                mid = identities.get(m)
                if mid:
                    inventory.update(mid.marker_genes)
            comm.marker_gene_inventory = sorted(inventory)
            comm.missing_markers = sorted(full_scg_set - inventory) if full_scg_set else []
            if full_scg_set:
                comm.completeness = round(100.0 * len(inventory) / len(full_scg_set), 2)
                c = comm.completeness
                comm.quality_tier = (
                    "excellent" if c >= 90 else
                    "high" if c >= 80 else
                    "good" if c >= 70 else
                    "fair" if c >= 50 else "low"
                )
        communities.update(new_comms)
        log(f"[INFO] Seeded {len(new_comms)} binner-agreement communities ({len(binner_assigned):,} contigs)")
    log(f"[INFO] Total: {len(identities):,} contigs, {len(communities)} communities")

    # Compute fit scores for binned contigs
    from nclb.valence import contig_fit_score as cv
    for name, c in identities.items():
        if c.community and c.community in communities:
            c.fit_score = cv(c, communities[c.community], adjacency)

    # Compute global fit score distribution for LLM context
    all_fit = [c.fit_score for c in identities.values()
               if c.community and c.fit_score is not None]
    if all_fit:
        all_fit_sorted = sorted(all_fit)
        n = len(all_fit_sorted)
        fit_dist = {
            "p25": all_fit_sorted[n // 4],
            "median": all_fit_sorted[n // 2],
            "p75": all_fit_sorted[3 * n // 4],
        }
        log(f"[INFO] Fit score distribution (n={n:,}): "
            f"p25={fit_dist['p25']:+.3f}, median={fit_dist['median']:+.3f}, "
            f"p75={fit_dist['p75']:+.3f}")
    else:
        fit_dist = None

    # Coherence metrics are computed lazily per-bin (JIT) to avoid 3+ min startup
    from nclb.valence import community_metrics
    from nclb.graph import graph_connectivity

    def ensure_coherence(comm):
        """Compute coherence metrics on first access."""
        if comm.tnf_coherence == 0.0 and len(comm.members) > 1:
            members = [identities[n] for n in comm.members if n in identities]
            comm.tnf_coherence, comm.tnf_sim_stdev = tnf_coherence(members)
            comm.coverage_correlation = coverage_coherence(members)
            comm.graph_connectivity = graph_connectivity(comm.members, adjacency)

    # Load landscape data (UMAP coordinates) if available
    landscape_path = binning_dir / "nclb" / "landscape.json"
    if landscape_path.exists():
        from nclb.landscape import load_landscape
        landscape_data = load_landscape(landscape_path)
        for name, lr in landscape_data.items():
            if name in identities:
                identities[name].landscape_x = lr.x
                identities[name].landscape_y = lr.y
                identities[name].landscape_cluster = lr.cluster
                identities[name].landscape_cluster_cx = lr.cluster_cx
                identities[name].landscape_cluster_cy = lr.cluster_cy
        log(f"[INFO] Loaded landscape data for {len(landscape_data):,} contigs")

    # Load gathering.json for pre-computed resonance and uneasy member data
    gathering = None
    if gathering_path.exists():
        with open(gathering_path) as f:
            gathering = json.load(f)
        log(f"[INFO] Loaded gathering data from {gathering_path}")

    # --- Load community display names (festive names) ---
    community_names: dict[str, str] = {}
    if gathering:
        community_names = gathering.get("community_names", {})
        if community_names:
            log(f"[INFO] Loaded {len(community_names)} festive community names")

    # --- Load Bakta annotations for toolkit ---
    bakta_annotations = load_bakta_annotations(bakta_path) if bakta_path.exists() else {}
    if bakta_annotations:
        log(f"[INFO] Loaded Bakta annotations for {len(bakta_annotations):,} contigs")

    # --- Create toolkit ---
    gfa_path = assembly_dir / "assembly_graph.gfa"
    toolkit = ContigToolkit(identities, communities, adjacency,
                            community_names=community_names,
                            annotations=bakta_annotations,
                            gfa_path=gfa_path if gfa_path.exists() else None)
    if toolkit._skeleton_gfa:
        skel_size = toolkit._skeleton_gfa.stat().st_size
        log(f"[INFO] Skeleton GFA: {toolkit._skeleton_gfa} ({skel_size:,} bytes)")

    # --- Configure OpenAI client ---
    from openai import OpenAI
    base_url = args.base_url
    model = args.model

    if not model:
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"{base_url}/models", timeout=5)
            models_data = json.loads(resp.read())
            if models_data.get("data"):
                model = models_data["data"][0]["id"]
                log(f"[INFO] Auto-detected model: {model}")
        except Exception as e:
            log(f"[WARNING] Could not auto-detect model: {e}")
            model = "local-model"

    client = OpenAI(base_url=base_url, api_key="lm-studio")
    log(f"[INFO] Server: {base_url}, Model: {model}")
    log("")

    all_proposals = []

    # =====================================================================
    # Round 1: Community Health Check (tool-use)
    # =====================================================================
    log("=" * 70)
    log("ROUND 1: COMMUNITY HEALTH CHECK")
    log("=" * 70)

    # Note: fit scores already computed at startup (line ~737) with current formula

    import random
    comm_order = list(communities.keys())
    random.shuffle(comm_order)
    for comm_name in comm_order:
        comm = communities[comm_name]

        # Re-read system prompt each iteration (hot-editable)
        r1_system = _load_prompt("round1_system.txt")
        if fit_dist:
            r1_system += (
                f"\n\nFit score IQR across all binned contigs: "
                f"[{fit_dist['p25']:+.3f}, {fit_dist['p75']:+.3f}]"
            )

        # JIT: compute coherence and build data dict per-bin (avoids 3+ min startup)
        ensure_coherence(comm)

        # Use pre-computed fit scores instead of recomputing via community_metrics
        member_scores = []
        for m in comm.members:
            c = identities.get(m)
            if c:
                member_scores.append((m, c.fit_score))
        fit_values = [s for _, s in member_scores]
        mean_fit = sum(fit_values) / len(fit_values) if fit_values else 0.0
        min_fit = min(fit_values) if fit_values else 0.0

        comm_data = {
            "name": comm_name,
            "quality_tier": comm.quality_tier,
            "members": comm.members,
            "total_size": comm.total_size,
            "completeness": comm.completeness,
            "redundancy": comm.redundancy,
            "tnf_coherence": comm.tnf_coherence,
            "coverage_correlation": comm.coverage_correlation,
            "graph_connectivity": comm.graph_connectivity,
            "mean_fit": mean_fit,
            "min_fit": min_fit,
        }

        prompt = round1_prompt(comm_name, comm_data, member_scores,
                               display_name=community_names.get(comm_name))
        try:
            result = run_tool_conversation(
                client, model, r1_system, prompt, toolkit,
                max_rounds=args.max_tool_rounds, log_fn=log,
            )
            n_releases = len(result.get("release", []))
            n_splits = len(result.get("split", []))
            n_tc = result.pop("_tool_calls", 0)
            parts = [f"{n_releases} releases"]
            if n_splits:
                parts.append(f"{n_splits} splits")
            log(f"  {comm_name}: {', '.join(parts)} ({n_tc} tool calls)")
            proposal = {"round": 1, "bin": comm_name, "result": result}
            all_proposals.append(proposal)
            with open(output_path.parent / "proposals.jsonl", "a") as jf:
                jf.write(json.dumps(proposal, default=str) + "\n")
        except Exception as e:
            log(f"  [ERROR] {comm_name}: {e}")

    r1_releases = sum(
        len(p.get("result", {}).get("release", []))
        for p in all_proposals if p["round"] == 1
    )
    r1_splits = sum(
        1 for p in all_proposals if p["round"] == 1
        and p.get("result", {}).get("split")
    )
    log(f"\nRound 1 complete: {r1_releases} releases, {r1_splits} splits from {len(communities)} communities")

    # Update landscape plot after Round 1
    if landscape_data:
        from nclb.landscape import plot_landscape
        plot_path = output_path.parent / "landscape.png"
        plot_landscape(landscape_data, identities, communities, plot_path,
                       community_names=community_names,
                       title="NCLB Contig Landscape — after Round 1")
        log(f"[INFO] Updated landscape plot → {plot_path}")

    # =====================================================================
    # Round 2: Unhoused Contigs Speak (tool-use)
    # =====================================================================
    log("")
    log("=" * 70)
    log("ROUND 2: UNHOUSED CONTIGS SPEAK")
    log("=" * 70)

    # Build per-contig candidate lookup from gathering.json resonance_candidates
    contig_candidates: dict[str, list[str]] = {}
    if gathering:
        for comm_name, candidates in gathering.get("resonance_candidates", {}).items():
            for cand in candidates:
                contig_candidates.setdefault(cand["contig"], []).append(comm_name)
        log(f"  Pre-computed candidates for {len(contig_candidates)} contigs from gathering.json")

    # Select unbinned contigs with binner support (sorted by n_binners, then size)
    unbinned = [
        c for c in identities.values()
        if c.community is None and c.n_binners >= 2
    ]
    unbinned.sort(key=lambda c: (-c.n_binners, -c.size))
    unbinned = unbinned[:args.max_round2]
    log(f"  Processing {len(unbinned)} unbinned contigs (n_binners >= 2)")

    # Batch contigs for conversation
    batches = []
    for i in range(0, len(unbinned), args.batch_size):
        batches.append([c.name for c in unbinned[i:i + args.batch_size]])

    for batch_idx, batch_names in enumerate(batches):
        prompt = round2_prompt(batch_names, contig_candidates, community_names)
        try:
            result = run_tool_conversation(
                client, model, _load_prompt("round2_system.txt"), prompt, toolkit,
                max_rounds=args.max_tool_rounds, log_fn=log,
            )
            n_tc = result.pop("_tool_calls", 0)
            # Translate festive display names back to internal names
            festive_to_internal = {v: k for k, v in community_names.items()}
            for d in result.get("decisions", []):
                if d.get("action") == "join" and d.get("bin"):
                    raw_name = d["bin"]
                    internal = festive_to_internal.get(raw_name)
                    if internal:
                        d["display_name"] = raw_name
                        d["bin"] = internal
                    elif raw_name not in communities:
                        # LLM invented a bin name — downgrade to "wait"
                        log(f"    [WARNING] '{raw_name}' is not a valid bin — converting join→wait for {d.get('contig', '?')}")
                        d["action"] = "wait"
                        d["bin"] = None
                        d["evidence"] = f"(original: join {raw_name}) {d.get('evidence', '')}"
            n_decisions = len(result.get("decisions", []))
            n_joins = sum(1 for d in result.get("decisions", []) if d.get("action") == "join")
            log(f"  Batch {batch_idx+1}/{len(batches)}: {n_decisions} decisions ({n_joins} joins, {n_tc} tool calls)")
            proposal = {"round": 2, "batch": batch_idx, "result": result}
            all_proposals.append(proposal)
            with open(output_path.parent / "proposals.jsonl", "a") as jf:
                jf.write(json.dumps(proposal, default=str) + "\n")
        except Exception as e:
            log(f"  [ERROR] Batch {batch_idx+1}: {e}")

    r2_decisions = sum(
        len(p.get("result", {}).get("decisions", []))
        for p in all_proposals if p["round"] == 2
    )
    r2_joins = sum(
        1 for p in all_proposals if p["round"] == 2
        for d in p.get("result", {}).get("decisions", [])
        if d.get("action") == "join"
    )
    log(f"\nRound 2 complete: {r2_decisions} decisions ({r2_joins} joins)")

    # Update landscape plot after Round 2
    if landscape_data:
        from nclb.landscape import plot_landscape
        plot_path = output_path.parent / "landscape.png"
        plot_landscape(landscape_data, identities, communities, plot_path,
                       community_names=community_names,
                       title="NCLB Contig Landscape — after Round 2")
        log(f"[INFO] Updated landscape plot → {plot_path}")

    # =====================================================================
    # Round 3: Voiceless Clusters
    # =====================================================================
    log("")
    log("=" * 70)
    log("ROUND 3: THE VOICELESS SPEAK")
    log("=" * 70)

    r3_proposals = []
    clusters = cluster_voiceless(identities)
    if clusters:
        log(f"  Found {len(clusters)} voiceless clusters")
        prompt = round3_prompt(clusters)
        try:
            result = run_tool_conversation(
                client, model, _load_prompt("round3_system.txt"), prompt, toolkit,
                max_rounds=3, log_fn=log,
            )
            result.pop("_tool_calls", None)
            proposal = {"round": 3, "clusters": clusters, "result": result}
            r3_proposals.append(proposal)
            with open(output_path.parent / "proposals.jsonl", "a") as jf:
                jf.write(json.dumps(proposal, default=str) + "\n")
        except Exception as e:
            log(f"  [ERROR] Round 3: {e}")
    else:
        log("  No viable voiceless clusters found")
    all_proposals.extend(r3_proposals)

    # Final landscape plot after Round 3
    if landscape_data:
        from nclb.landscape import plot_landscape
        plot_path = output_path.parent / "landscape.png"
        plot_landscape(landscape_data, identities, communities, plot_path,
                       community_names=community_names,
                       title="NCLB Contig Landscape — final")
        log(f"[INFO] Updated landscape plot → {plot_path}")

    # --- Summary ---
    summary = {
        "round1": {"n_communities": len(communities), "n_releases": r1_releases, "n_splits": r1_splits},
        "round2": {"n_batches": len(batches), "n_decisions": r2_decisions, "n_joins": r2_joins},
        "round3": {
            "n_clusters": sum(len(p.get("clusters", [])) for p in r3_proposals),
            "n_accepted": sum(
                1 for p in r3_proposals
                for e in p.get("result", {}).get("evaluations", [])
                if e.get("verdict") == "accept"
            ),
        },
    }

    # --- Write output ---
    output = {
        "version": "0.3.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": "openai",
        "model": model,
        "architecture": "tool-use",
        "summary": summary,
        "proposals": all_proposals,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0

    log("")
    log("=" * 70)
    log("GATHERING 2: COMPLETE")
    log("=" * 70)
    log(f"  Round 1: {r1_releases} releases, {r1_splits} splits")
    log(f"  Round 2: {r2_joins} joins out of {r2_decisions} decisions")
    log(f"  Round 3: {summary['round3']['n_accepted']} new communities")
    log(f"  Output: {output_path}")
    log(f"  Time:   {elapsed:.1f}s")
    log("=" * 70)


if __name__ == "__main__":
    main()
