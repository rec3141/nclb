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

ROUND1_SYSTEM = """You examine metagenome-assembled genome bins for quality issues.

Each bin contains contigs placed together by consensus binning.
You examine each bin's coherence and identify contigs that don't belong.

Available tools:
- get_contig_info(contig): size, GC%, coverage, taxonomy, domain (prokaryotic/eukaryotic/organellar), gene names, marker_genes (SCGs), n_cds, MGE status (viral/plasmid/provirus), defense systems, integrons, secretion systems, genomic islands, binner agreement (n_binners)
- get_graph_neighbors(contig): assembly graph neighbors with their bin assignments
- get_binner_assignments(contig): per-binner assignments, agreement count, consensus placement
- compare_to_bin(contig, community): fit score, TNF cosine similarity, coverage Pearson r, graph neighbor fraction, GC comparison
- get_bin_info(community): members, total size, GC range, completeness, contamination, coverage profile, coherence metrics, quality tier, marker gene inventory
- get_missing_markers(community): marker genes the bin still needs for completeness
- predict_join_impact(contig, community): predicted size/GC shift, new marker gene contributions
- find_graph_connections(contig): which bins this contig connects to via assembly graph
- read_annotations(contig, page=1): paginated CDS annotation table (20 features per page)

Actions you can recommend:
- RELEASE a contig: remove it from this community (it becomes unbinned). Only release
  contigs that clearly do not belong — e.g. different taxonomy, wildly different GC/coverage,
  no graph connections, low binner agreement. Strong binner agreement (n_binners >= 3) is evidence
  a contig BELONGS, not a reason to release it.
- SPLIT the community: if you find the community contains two or more distinct groups
  (different phyla, divergent coverage patterns, bimodal GC), recommend splitting it.
  List which contigs belong to each proposed sub-group.

Investigate thoroughly, weigh the evidence, and make your best judgment.

After investigating, respond with JSON (no commentary):
{"community": "name", "assessment": "narrative", "release": [{"contig": "name", "reason": "evidence"}], "split": [{"name": "descriptive label", "members": ["contig1", "contig2"]}], "concerns": []}"""

ROUND2_SYSTEM = """You evaluate unbinned contigs seeking bin placement.

You have tools to investigate each contig's identity, graph connections,
and compatibility with candidate bins. Call them to build evidence.

Available tools:
- get_contig_info(contig): size, GC%, coverage, taxonomy, domain (prokaryotic/eukaryotic/organellar), gene names, marker_genes (SCGs), n_cds, MGE status (viral/plasmid/provirus), defense systems, integrons, secretion systems, genomic islands, binner agreement (n_binners)
- get_graph_neighbors(contig): assembly graph neighbors with their bin assignments
- get_binner_assignments(contig): per-binner assignments, agreement count, consensus placement
- compare_to_bin(contig, community): fit score, TNF cosine similarity, coverage Pearson r, graph neighbor fraction, GC comparison
- get_bin_info(community): members, total size, GC range, completeness, contamination, coverage profile, coherence metrics, quality tier, marker gene inventory
- get_missing_markers(community): marker genes the bin still needs for completeness
- predict_join_impact(contig, community): predicted size/GC shift, new marker gene contributions
- find_graph_connections(contig): which bins this contig connects to via assembly graph
- read_annotations(contig, page=1): paginated CDS annotation table (20 features per page)

These contigs were recognized by binning algorithms but not placed in the
consensus. Investigate each and find where they belong.

CRITICAL: Only use community names that appear in your prompt's candidate list or
that are returned by find_graph_connections() / get_graph_neighbors() tool calls.
The binner_assignments field in get_contig_info() shows per-binner bin IDs (e.g. "semibin_074")
which are NOT valid community names — never use binner IDs as community names.
Never invent or guess community names. If no valid community is found, use "wait".

After investigating, respond with JSON (no commentary):
{"decisions": [{"contig": "name", "action": "join|wait|wander", "community": "name_or_null", "evidence": "specific signals", "fit_score": 0.0}]}"""

ROUND3_SYSTEM = """You evaluate candidate new communities formed from unbinned contigs.

Signs of a real community: TNF coherence >0.9, synchronized coverage across
samples, reasonable genome size, consistent ancestry.

Respond with JSON (no commentary):
{"evaluations": [{"cluster_id": 0, "verdict": "accept|reject|uncertain", "reason": "assessment", "suggested_name": "name"}]}"""


# ---------------------------------------------------------------------------
# Tool-use conversation loop
# ---------------------------------------------------------------------------

def _summarize_tool_result(tool_name: str, result: dict) -> str:
    """Produce a short human-readable summary of a tool result for logging."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    if tool_name == "get_contig_info":
        parts = [f"{result.get('size', 0):,}bp"]
        if result.get("ancestry"):
            # Show last two ranks of lineage
            lineage = result["ancestry"].split(";")
            parts.append(lineage[-1].strip() if lineage else "?")
        if result.get("mge_type"):
            parts.append(result["mge_type"])
        if result.get("gene_names"):
            parts.append(f"{len(result['gene_names'])} genes")
        if result.get("marker_genes"):
            parts.append(f"{len(result['marker_genes'])} SCGs")
        if result.get("has_defense_system"):
            n = len(result.get("defense_systems", []))
            parts.append(f"{n} defense")
        if result.get("has_integron"):
            parts.append("integron")
        if result.get("has_secretion_system"):
            parts.append("secretion")
        domain = result.get("domain", "unknown")
        if domain != "unknown":
            parts.append(domain)
        n_cds = result.get("n_cds", 0)
        if n_cds:
            parts.append(f"{n_cds} CDS")
        parts.append(f"n_binners={result.get('n_binners', 0)}")
        return ", ".join(parts)

    if tool_name == "get_binner_assignments":
        binner_assignments = result.get("binner_assignments", {})
        assigned = [f"{k}={v}" for k, v in binner_assignments.items() if v]
        return f"n_binners={result.get('n_binners', 0)}, {', '.join(assigned) or 'none'}"

    if tool_name == "compare_to_bin":
        return (
            f"fit_score={result.get('fit_score', 0):+.3f}, "
            f"tnf_cosine={result.get('tnf_cosine_similarity', 0):.3f}, "
            f"cov_pearson_r={result.get('cov_pearson_r', 0):.3f}, "
            f"graph_neighbor_frac={result.get('graph_neighbor_fraction', 0):.3f}, "
            f"GC_delta={result.get('gc_delta', 0):.4f}"
        )

    if tool_name == "get_bin_info":
        return (
            f"{result.get('n_members', 0)} members, "
            f"{result.get('total_size', 0):,}bp, "
            f"{result.get('completeness', 0):.0f}% complete, "
            f"{result.get('quality_tier', '?')}, "
            f"{len(result.get('missing_markers', []))} missing SCGs"
        )

    if tool_name == "get_missing_markers":
        return f"{result.get('n_missing', 0)} missing markers"

    if tool_name == "predict_join_impact":
        return (
            f"+{result.get('size_delta', 0):,}bp, "
            f"GC_shift={result.get('gc_shift', 0):.4f}, "
            f"{result.get('n_contributed', 0)} contributed SCGs"
        )

    if tool_name == "find_graph_connections":
        conns = result.get("community_connections", [])
        if conns:
            top = ", ".join(f"{c['community']}({c['n_edges']})" for c in conns[:3])
            return f"{len(conns)} communities: {top}"
        return "no graph connections"

    if tool_name == "get_graph_neighbors":
        neighbors = result.get("neighbors", [])
        housed = sum(1 for n in neighbors if n.get("community"))
        return f"{len(neighbors)} neighbors ({housed} housed)"

    if tool_name == "read_annotations":
        return (
            f"page {result.get('page', 1)}/{result.get('total_pages', 1)}, "
            f"{result.get('total_features', 0)} total features"
        )

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

def round1_prompt(comm_name: str, comm_data: dict, uneasy_names: list[str],
                  display_name: str | None = None) -> str:
    """Brief prompt for Round 1 — the LLM investigates via tools."""
    uneasy_section = "None — all members have positive fit score."
    if uneasy_names:
        uneasy_section = ", ".join(uneasy_names)

    shown_name = display_name or comm_name
    quality = comm_data.get('quality_tier', 'none')
    return f"""Examine community "{shown_name}".

Quick overview:
  Quality tier: {quality}
  Members: {len(comm_data.get('members', []))}
  Size: {comm_data['total_size']:,} bp
  Wholeness: {comm_data['completeness']:.1f}% | Redundancy: {comm_data['redundancy']:.1f}%
  Mean fit: {comm_data['mean_fit']:+.3f} | Min fit: {comm_data['min_fit']:+.3f}

Low-fit members (negative fit score): {uneasy_section}

Investigate each low-fit member using get_contig_info() and compare_to_bin().
Check their coverage, graph connections, and binner assignments.
Then decide which should be released."""


def round2_prompt(
    contig_names: list[str],
    contig_candidates: dict[str, list[str]] | None = None,
    community_names: dict[str, str] | None = None,
) -> str:
    """Brief prompt for Round 2 — the LLM investigates via tools."""
    display = community_names or {}
    lines = []
    for name in contig_names:
        candidates = contig_candidates.get(name, []) if contig_candidates else []
        if candidates:
            display_names = [display.get(c, c) for c in candidates[:5]]
            lines.append(f"  {name}: candidates → {', '.join(display_names)}")
        else:
            lines.append(f"  {name}: no pre-computed candidates (use find_graph_connections)")

    contig_section = "\n".join(lines)
    return f"""You evaluate {len(contig_names)} unbinned contigs seeking bin placement.

Contigs and their candidate communities:
{contig_section}

For each contig:
1. Call get_contig_info() to learn its full identity
2. Call compare_to_bin(contig, community) for the candidate communities listed above
3. If no candidates are listed, call find_graph_connections() to discover communities
4. Decide: join (specify the community name exactly as listed), wait, or wander

IMPORTANT: Only use community names from the candidate list above or from tool results.
Binner bin IDs like "semibin_074" are NOT community names. Never invent names.

Investigate each contig and recommend placement."""


def round3_prompt(clusters: list[dict]) -> str:
    """Prompt for Round 3 — data-in-prompt is fine for cluster summaries."""
    lines = []
    for cl in clusters:
        lines.append(
            f"  Cluster {cl['id']}: {cl['n_members']} contigs, "
            f"{cl['total_size']:,}bp, TNF coherence={cl['tnf_coherence']:.3f}, "
            f"Coverage correlation={cl.get('coverage_correlation', 0):.3f}, "
            f"Mean GC={cl.get('mean_gc', 0):.3f}"
        )
    return f"""{len(clusters)} clusters emerged from voiceless contigs.

CANDIDATES:
{chr(10).join(lines)}

Evaluate each: real organism or noise?"""


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
            "tnf_coherence": float(tnf_coherence(members)),
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
        defensefinder_path=df_path if df_path.exists() else None,
        prokka_gff_path=prokka_gff_path,
        bacteria_scg_path=bacteria_scg if bacteria_scg.exists() else None,
        archaea_scg_path=archaea_scg if archaea_scg.exists() else None,
        eukaryotic_path=eukaryotic_path if eukaryotic_path.exists() else None,
        bakta_tsv_path=bakta_path if bakta_path.exists() else None,
    )
    gfa_adjacency = load_gfa_graph(assembly_dir / "assembly_graph.gfa")

    # Load read-bridged adjacency from BAMs (supplementary alignments)
    bam_files = sorted(mapping_dir.glob("*.sorted.bam"))
    if bam_files:
        read_adj = load_read_adjacency(bam_files)
        adjacency = merge_adjacencies(gfa_adjacency, read_adj)
        log(f"[INFO] Adjacency: GFA={len(gfa_adjacency)} contigs, "
            f"read-bridged={len(read_adj)} contigs, "
            f"merged={len(adjacency)} contigs")
        # Update contig identity connections with merged adjacency
        for name, c in identities.items():
            c.connections = adjacency.get(name, [])
    else:
        adjacency = gfa_adjacency
        log(f"[INFO] No BAM files found in {mapping_dir}, using GFA adjacency only")

    log(f"[INFO] Loaded {len(identities):,} contigs, {len(communities)} DAS Tool communities")

    # Log eukaryotic classification stats
    n_euk = sum(1 for c in identities.values() if c.domain_class == "eukaryotic")
    n_org = sum(1 for c in identities.values() if c.domain_class == "organellar")
    if n_euk or n_org:
        log(f"[INFO] Domain classification: {n_euk} eukaryotic, {n_org} organellar contigs")

    # Seed additional communities from binner agreement
    from nclb.identity import seed_communities_from_binner_agreement
    new_comms, binner_assigned = seed_communities_from_binner_agreement(identities)
    if new_comms:
        for contig, comm_name in binner_assigned.items():
            identities[contig].community = comm_name
            identities[contig].membership_type = "core"
        communities.update(new_comms)
        log(f"[INFO] Seeded {len(new_comms)} binner-agreement communities ({len(binner_assigned):,} contigs)")
    log(f"[INFO] Total: {len(identities):,} contigs, {len(communities)} communities")

    # Compute fit scores for housed contigs
    from nclb.valence import contig_fit_score as cv
    for name, c in identities.items():
        if c.community and c.community in communities:
            c.fit_score = cv(c, communities[c.community], adjacency)

    # Compute community coherence metrics
    from nclb.valence import community_metrics
    from nclb.graph import graph_connectivity
    for comm in communities.values():
        members = [identities[n] for n in comm.members if n in identities]
        comm.tnf_coherence = tnf_coherence(members)
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
    toolkit = ContigToolkit(identities, communities, adjacency,
                            community_names=community_names,
                            annotations=bakta_annotations)

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

    # Build community data dicts for prompts
    comm_data_map = {}
    uneasy_map = {}
    for comm_name, comm in communities.items():
        harmony_report = community_metrics(comm, identities, adjacency)
        comm_data_map[comm_name] = {
            "name": comm_name,
            "quality_tier": comm.quality_tier,
            "members": comm.members,
            "total_size": comm.total_size,
            "completeness": comm.completeness,
            "redundancy": comm.redundancy,
            "tnf_coherence": comm.tnf_coherence,
            "coverage_correlation": comm.coverage_correlation,
            "graph_connectivity": comm.graph_connectivity,
            "mean_fit": harmony_report["mean_fit"],
            "min_fit": harmony_report["min_fit"],
        }
        # Find uneasy members
        uneasy_names = []
        for m in comm.members:
            c = identities.get(m)
            if c and c.fit_score < 0:
                uneasy_names.append(m)
        uneasy_map[comm_name] = uneasy_names

    for comm_name in sorted(communities.keys()):
        uneasy = uneasy_map[comm_name]
        comm_data = comm_data_map[comm_name]

        if not uneasy:
            log(f"  {comm_name}: 0 uneasy, skipping")
            all_proposals.append({
                "round": 1, "community": comm_name,
                "result": {"community": comm_name, "assessment": "All members content", "release": []},
            })
            continue

        prompt = round1_prompt(comm_name, comm_data, uneasy,
                               display_name=community_names.get(comm_name))
        try:
            result = run_tool_conversation(
                client, model, ROUND1_SYSTEM, prompt, toolkit,
                max_rounds=args.max_tool_rounds, log_fn=log,
            )
            n_releases = len(result.get("release", []))
            n_splits = len(result.get("split", []))
            n_tc = result.pop("_tool_calls", 0)
            parts = [f"{n_releases} releases"]
            if n_splits:
                parts.append(f"{n_splits} splits")
            log(f"  {comm_name}: {', '.join(parts)} ({n_tc} tool calls)")
            all_proposals.append({
                "round": 1, "community": comm_name, "result": result,
            })
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

    # Select unhoused contigs with binner support (sorted by n_binners, then size)
    unhoused = [
        c for c in identities.values()
        if c.community is None and c.n_binners >= 2
    ]
    unhoused.sort(key=lambda c: (-c.n_binners, -c.size))
    unhoused = unhoused[:args.max_round2]
    log(f"  Processing {len(unhoused)} unhoused contigs (n_binners >= 2)")

    # Batch contigs for conversation
    batches = []
    for i in range(0, len(unhoused), args.batch_size):
        batches.append([c.name for c in unhoused[i:i + args.batch_size]])

    for batch_idx, batch_names in enumerate(batches):
        prompt = round2_prompt(batch_names, contig_candidates, community_names)
        try:
            result = run_tool_conversation(
                client, model, ROUND2_SYSTEM, prompt, toolkit,
                max_rounds=args.max_tool_rounds, log_fn=log,
            )
            n_tc = result.pop("_tool_calls", 0)
            # Translate festive display names back to internal names
            festive_to_internal = {v: k for k, v in community_names.items()}
            for d in result.get("decisions", []):
                if d.get("action") == "join" and d.get("community"):
                    raw_name = d["community"]
                    internal = festive_to_internal.get(raw_name)
                    if internal:
                        d["display_name"] = raw_name
                        d["community"] = internal
                    elif raw_name not in communities:
                        # LLM invented a community name — downgrade to "wait"
                        log(f"    [WARNING] '{raw_name}' is not a valid community — converting join→wait for {d.get('contig', '?')}")
                        d["action"] = "wait"
                        d["community"] = None
                        d["evidence"] = f"(original: join {raw_name}) {d.get('evidence', '')}"
            n_decisions = len(result.get("decisions", []))
            n_joins = sum(1 for d in result.get("decisions", []) if d.get("action") == "join")
            log(f"  Batch {batch_idx+1}/{len(batches)}: {n_decisions} decisions ({n_joins} joins, {n_tc} tool calls)")
            all_proposals.append({
                "round": 2, "batch": batch_idx, "result": result,
            })
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
                client, model, ROUND3_SYSTEM, prompt, toolkit,
                max_rounds=3, log_fn=log,
            )
            result.pop("_tool_calls", None)
            r3_proposals.append({"round": 3, "clusters": clusters, "result": result})
        except Exception as e:
            log(f"  [ERROR] Round 3: {e}")
    else:
        log("  No viable voiceless clusters found")
    all_proposals.extend(r3_proposals)

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
