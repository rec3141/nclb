"""Voices — LLM prompt templates and contig-agent tool definitions.

The contigs are represented by LLM agents. Each conversation round has prompt
templates and tool definitions that let the LLM investigate on behalf
of contigs and communities.

Model hierarchy (the wisdom of scale):
  Haiku   — contig agents (Round 2: unhoused contigs, Round 3: voiceless clusters)
  Sonnet  — Elder agents (Round 1: community health, Elder investigations)
  Opus    — Contigsattva agents (mentoring, dispute mediation, Mediator)
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr

from .identity import ContigIdentity, CommunityProfile
from .valence import contig_fit_score, tnf_coherence, coverage_coherence
from .graph import graph_connectivity, find_graph_bridges


# ---------------------------------------------------------------------------
# Model hierarchy: the wisdom of scale
# ---------------------------------------------------------------------------

MODEL_TIERS = {
    "contig": "claude-haiku-4-5-20251001",       # fast, cheap — for 28K contigs
    "community": "claude-sonnet-4-5-20250929",   # balanced — for community-level reasoning
    "excellent": "claude-opus-4-6",              # deep — for high-quality bins + mediation
    "mediator": "claude-opus-4-6",               # conflict resolution
}


def model_for_role(role: str) -> str:
    """Get the appropriate Claude model for a given role."""
    return MODEL_TIERS.get(role, MODEL_TIERS["community"])


# ---------------------------------------------------------------------------
# Tool runtime: functions the LLM can call during conversations
# ---------------------------------------------------------------------------

class ContigToolkit:
    """Runtime for contig-agent tools.

    The LLM calls these via tool-use. Each returns a JSON-safe dict.
    """

    def __init__(
        self,
        identities: dict[str, ContigIdentity],
        communities: dict[str, CommunityProfile],
        adjacency: dict[str, list[str]],
        resonance_map=None,
        community_names: dict[str, str] | None = None,
        annotations: dict[str, list[dict]] | None = None,
    ):
        self.identities = identities
        self.communities = communities
        self.adjacency = adjacency
        self.resonance_map = resonance_map
        self.annotations = annotations or {}
        # community_names: internal_name → festive display name
        self._to_display = community_names or {}
        self._from_display = {v: k for k, v in self._to_display.items()}

    def _resolve(self, name: str) -> str:
        """Resolve a display name (or internal name) to internal name."""
        return self._from_display.get(name, name)

    def _display(self, name: str | None) -> str | None:
        """Convert internal community name to display name."""
        if name is None:
            return None
        return self._to_display.get(name, name)

    def get_contig_info(self, contig_name: str) -> dict:
        """Full identity card for a contig."""
        c = self.identities.get(contig_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        d = {
            "name": c.name,
            "size": c.size,
            "gc": round(c.gc, 4),
            "assembly_coverage": round(c.assembly_coverage, 2),
            "is_circular": c.is_circular,
            "is_repeat": c.is_repeat,
            "multiplicity": c.multiplicity,
            "coverage_per_sample": [round(float(x), 6) for x in c.coverage],
            "coverage_summary": {
                "n_samples": len(c.coverage),
                "n_detected": sum(1 for x in c.coverage if x > 0),
                "mean_nonzero": round(float(np.mean([x for x in c.coverage if x > 0])), 4) if any(x > 0 for x in c.coverage) else 0.0,
                "max": round(float(max(c.coverage)), 4),
            },
            "graph_neighbors": c.connections,
            "n_graph_neighbors": len(c.connections),
            "binner_assignments": c.binner_assignments,
            "n_binners": c.n_binners,
            "bin": self._display(c.community),
            "membership_type": c.membership_type,
            "ancestry": c.ancestry,
            "gene_names": c.gene_names,
            "marker_genes": c.marker_genes,
        }
        # Landscape position (UMAP coordinates + cluster centroid)
        if c.landscape_x != 0.0 or c.landscape_y != 0.0:
            d["position"] = [round(c.landscape_x, 2), round(c.landscape_y, 2)]
            d["landscape_cluster"] = c.landscape_cluster if c.landscape_cluster >= 0 else None
            if c.landscape_cluster >= 0:
                d["cluster_centroid"] = [round(c.landscape_cluster_cx, 2),
                                         round(c.landscape_cluster_cy, 2)]
        # MGE annotations — always present so LLM sees explicit negatives
        d["mge_type"] = c.mge_type or None
        d["is_viral"] = c.is_viral
        d["virus_score"] = round(c.virus_score, 4) if c.is_viral else None
        d["virus_hallmarks"] = c.virus_hallmarks if c.is_viral else None
        d["virus_taxonomy"] = c.virus_taxonomy or None
        d["is_plasmid"] = c.is_plasmid
        d["plasmid_score"] = round(c.plasmid_score, 4) if c.is_plasmid else None
        d["plasmid_hallmarks"] = c.plasmid_hallmarks if c.is_plasmid else None
        d["conjugation_genes"] = c.conjugation_genes or []
        d["amr_genes"] = c.amr_genes or []
        d["is_provirus"] = c.is_provirus
        d["proviral_length"] = c.proviral_length if c.is_provirus else None
        d["checkv_quality"] = c.checkv_quality or None
        d["viral_genes"] = c.viral_genes
        d["host_genes"] = c.host_genes
        # HGT / defense — always present
        d["has_integron"] = c.has_integron
        d["integrons"] = c.integrons or []
        d["has_genomic_island"] = c.has_genomic_island
        d["genomic_islands"] = c.genomic_islands or []
        d["has_secretion_system"] = c.has_secretion_system
        d["secretion_systems"] = c.secretion_systems or []
        d["has_defense_system"] = c.has_defense_system
        d["defense_systems"] = c.defense_systems or []
        d["coding_density"] = round(c.coding_density, 4) if c.coding_density else None
        # Domain classification
        d["domain"] = c.domain_class
        d["domain_confidence"] = c.domain_confidence
        d["organellar_subtype"] = c.organellar_subtype
        # Annotation summary
        d["n_cds"] = len(self.annotations.get(c.name, []))
        return d

    def get_graph_neighbors(self, contig_name: str) -> dict:
        """Assembly graph neighbors with their bin assignments."""
        c = self.identities.get(contig_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        neighbors = []
        for n in c.connections:
            nc = self.identities.get(n)
            if nc:
                neighbors.append({
                    "name": n,
                    "size": nc.size,
                    "bin": self._display(nc.community),
                    "gc": round(nc.gc, 4),
                })
        return {"contig": contig_name, "neighbors": neighbors}

    def get_binner_assignments(self, contig_name: str) -> dict:
        """All 5 binner assignments for a contig."""
        c = self.identities.get(contig_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        return {
            "contig": contig_name,
            "binner_assignments": c.binner_assignments,
            "n_binners": c.n_binners,
            "consensus": self._display(c.community),
        }

    def compare_to_bin(self, contig_name: str, bin_name: str) -> dict:
        """Compute fit score of a contig against a specific bin.

        Returns both computed metrics AND raw data for the LLM to assess.
        """
        bin_name = self._resolve(bin_name)
        c = self.identities.get(contig_name)
        comm = self.communities.get(bin_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        if not comm:
            return {"error": f"Unknown bin: {bin_name}"}

        v = contig_fit_score(c, comm, self.adjacency)

        # Individual signal components
        tnf_cos = 0.0
        if comm.tnf_centroid is not None and c.tnf is not None:
            tnf_cos = 1.0 - cosine_distance(c.tnf, comm.tnf_centroid)

        cov_r = 0.0
        if comm.mean_coverage is not None and c.coverage is not None:
            if len(c.coverage) > 1 and np.std(c.coverage) > 0 and np.std(comm.mean_coverage) > 0:
                r, _ = pearsonr(c.coverage, comm.mean_coverage)
                cov_r = max(0.0, r)

        member_set = set(comm.members)
        neighbors_in = [n for n in c.connections if n in member_set]
        neighbor_frac = len(neighbors_in) / len(c.connections) if c.connections else 0.0

        return {
            "contig": contig_name,
            "bin": self._display(bin_name),
            "fit_score": round(v, 4),
            "tnf_cosine_similarity": round(tnf_cos, 4),
            "cov_pearson_r": round(cov_r, 4),
            "graph_neighbor_fraction": round(neighbor_frac, 4),
            "graph_neighbors_in_community": neighbors_in,
            "n_graph_neighbors_in": len(neighbors_in),
            "contig_coverage": [round(float(x), 6) for x in c.coverage],
            "bin_mean_coverage": [round(float(x), 6) for x in comm.mean_coverage] if comm.mean_coverage is not None else [],
            "coverage_pattern_match": (
                f"both detected in same {sum(1 for a, b in zip(c.coverage, comm.mean_coverage) if a > 0 and b > 0)} samples"
                if comm.mean_coverage is not None and sum(1 for a, b in zip(c.coverage, comm.mean_coverage) if (a > 0) == (b > 0)) == len(c.coverage)
                else "different detection patterns"
            ) if comm.mean_coverage is not None else "no bin coverage data",
            "contig_gc": round(c.gc, 4),
            "bin_mean_gc": round(comm.mean_gc, 4),
            "gc_delta": round(abs(c.gc - comm.mean_gc), 4),
            "bin_completeness": round(comm.completeness, 2),
            "bin_quality_tier": comm.quality_tier,
        }

    def get_bin_info(self, bin_name: str) -> dict:
        """Full bin profile."""
        bin_name = self._resolve(bin_name)
        comm = self.communities.get(bin_name)
        if not comm:
            return {"error": f"Unknown bin: {bin_name}"}
        d = {
            "name": self._display(comm.name),
            "source_binner": comm.source_binner,
            "n_members": len(comm.members),
            "total_size": comm.total_size,
            "n50": comm.n50,
            "mean_gc": round(comm.mean_gc, 4),
            "gc_stdev": round(comm.gc_stdev, 4),
            "completeness": round(comm.completeness, 2),
            "redundancy": round(comm.redundancy, 2),
            "quality_tier": comm.quality_tier,
            "tnf_coherence": round(comm.tnf_coherence, 4),
            "coverage_correlation": round(comm.coverage_correlation, 4),
            "graph_connectivity": round(comm.graph_connectivity, 4),
            "missing_markers": comm.missing_markers,
        }
        # Include mean coverage profile (actual data)
        if comm.mean_coverage is not None:
            d["mean_coverage_per_sample"] = [round(float(x), 6) for x in comm.mean_coverage]
        return d

    def get_missing_markers(self, bin_name: str) -> dict:
        """Marker genes the bin still needs."""
        bin_name = self._resolve(bin_name)
        comm = self.communities.get(bin_name)
        if not comm:
            return {"error": f"Unknown bin: {bin_name}"}
        return {
            "bin": self._display(bin_name),
            "completeness": round(comm.completeness, 2),
            "missing_markers": comm.missing_markers,
            "n_missing": len(comm.missing_markers),
        }

    def predict_join_impact(self, contig_name: str, bin_name: str) -> dict:
        """Predict impact of adding a contig to a bin."""
        bin_name = self._resolve(bin_name)
        c = self.identities.get(contig_name)
        comm = self.communities.get(bin_name)
        if not c or not comm:
            return {"error": "Unknown contig or bin"}

        # Size change
        new_size = comm.total_size + c.size

        # New mean GC
        member_gcs = [self.identities[n].gc for n in comm.members if n in self.identities]
        old_mean_gc = np.mean(member_gcs) if member_gcs else 0.0
        new_mean_gc = np.mean(member_gcs + [c.gc]) if member_gcs else c.gc

        # Contribution: marker genes the contig carries that community lacks
        contributed_markers = []
        if c.marker_genes and comm.missing_markers:
            contributed_markers = list(set(c.marker_genes) & set(comm.missing_markers))

        # Would any existing member's fit score drop?
        # (approximate: check if new centroid shifts significantly)
        gc_shift = abs(new_mean_gc - old_mean_gc)

        return {
            "contig": contig_name,
            "bin": self._display(bin_name),
            "size_delta": c.size,
            "new_total_size": new_size,
            "gc_shift": round(gc_shift, 4),
            "contributed_markers": contributed_markers,
            "n_contributed": len(contributed_markers),
        }

    def find_similar_contigs(self, contig_name: str, k: int = 10) -> dict:
        """K nearest contigs by TNF composition."""
        if self.resonance_map is None:
            return {"error": "Resonance map not available"}
        neighbors = self.resonance_map.find_nearest_contigs(contig_name, k)
        return {
            "contig": contig_name,
            "nearest": [
                {
                    "name": name,
                    "similarity": round(sim, 4),
                    "bin": self._display(self.identities[name].community) if name in self.identities else None,
                }
                for name, sim in neighbors
            ],
        }

    def find_graph_connections(self, contig_name: str) -> dict:
        """Which bins is a contig graph-connected to?"""
        bridges = find_graph_bridges(contig_name, self.communities, self.adjacency)
        return {
            "contig": contig_name,
            "bin_connections": [
                {"bin": self._display(name), "n_edges": n}
                for name, n in sorted(bridges.items(), key=lambda x: -x[1])
            ],
        }

    def read_annotations(self, contig_name: str, page: int = 1) -> dict:
        """Paginated CDS annotation table for a contig."""
        features = self.annotations.get(contig_name, [])
        if not features:
            c = self.identities.get(contig_name)
            if not c:
                return {"error": f"Unknown contig: {contig_name}"}
            return {
                "contig": contig_name,
                "total_features": 0,
                "page": 1,
                "total_pages": 1,
                "features": [],
            }
        page_size = 20
        total_pages = max(1, (len(features) + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        return {
            "contig": contig_name,
            "total_features": len(features),
            "page": page,
            "total_pages": total_pages,
            "features": features[start:start + page_size],
        }

    def dispatch(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch a tool call from the LLM."""
        dispatch_map = {
            "get_contig_info": self.get_contig_info,
            "get_graph_neighbors": self.get_graph_neighbors,
            "get_binner_assignments": self.get_binner_assignments,
            "compare_to_bin": self.compare_to_bin,
            "get_bin_info": self.get_bin_info,
            "get_missing_markers": self.get_missing_markers,
            "predict_join_impact": self.predict_join_impact,
            "find_similar_contigs": self.find_similar_contigs,
            "find_graph_connections": self.find_graph_connections,
            "read_annotations": self.read_annotations,
        }

        fn = dispatch_map.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}

        return fn(**arguments)


# ---------------------------------------------------------------------------
# Claude API tool definitions
# ---------------------------------------------------------------------------

CONTIG_TOOLS_ANTHROPIC = [
    {
        "name": "get_contig_info",
        "description": "Returns contig metadata: size, GC%, coverage, taxonomy, domain, gene names, marker genes, MGE status, consensus bin assignment, n_binners (how many individual tools binned it anywhere — not specific to any consensus bin).",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
            },
            "required": ["contig_name"],
        },
    },
    {
        "name": "get_graph_neighbors",
        "description": "Returns assembly graph neighbors of a contig with their bin assignments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
            },
            "required": ["contig_name"],
        },
    },
    {
        "name": "get_binner_assignments",
        "description": "Returns per-binner bin IDs for a contig (e.g. semibin_022, metabat_014). These are raw assignments from individual tools — each binner groups contigs independently, so per-binner IDs do NOT correspond to consensus bin names.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
            },
            "required": ["contig_name"],
        },
    },
    {
        "name": "compare_to_bin",
        "description": "Computes fit score between a contig and a consensus bin. Returns TNF cosine similarity, coverage Pearson r, graph neighbor fraction, and GC comparison. Use this to evaluate whether a contig belongs in a specific bin.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
                "bin_name": {"type": "string", "description": "Name of the bin"},
            },
            "required": ["contig_name", "bin_name"],
        },
    },
    {
        "name": "get_bin_info",
        "description": "Returns bin profile: members, size, completeness, coverage, coherence metrics, quality tier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bin_name": {"type": "string", "description": "Name of the bin"},
            },
            "required": ["bin_name"],
        },
    },
    {
        "name": "get_missing_markers",
        "description": "Lists marker genes a bin still needs for completeness.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bin_name": {"type": "string", "description": "Name of the bin"},
            },
            "required": ["bin_name"],
        },
    },
    {
        "name": "predict_join_impact",
        "description": "Predicts the impact of adding a contig to a bin: size change, GC shift, new marker gene contributions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
                "bin_name": {"type": "string", "description": "Name of the bin"},
            },
            "required": ["contig_name", "bin_name"],
        },
    },
    {
        "name": "find_graph_connections",
        "description": "Finds which bins a contig is connected to via the assembly graph.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
            },
            "required": ["contig_name"],
        },
    },
    {
        "name": "read_annotations",
        "description": "Returns paginated CDS annotation table for a contig — gene name, product, start/stop, strand, KEGG/EC/Pfam cross-references. 20 features per page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
                "page": {"type": "integer", "description": "Page number (default 1)", "default": 1},
            },
            "required": ["contig_name"],
        },
    },
]

# OpenAI-compatible format (for LM Studio, ollama, vLLM, etc.)
CONTIG_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in CONTIG_TOOLS_ANTHROPIC
]

# Backward compat
CONTIG_TOOLS = CONTIG_TOOLS_ANTHROPIC


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ROUND1_SYSTEM = """You evaluate metagenome-assembled genome bins.

Each bin is a group of contigs that were placed together by consensus binning.
You examine each bin's collective state — its coherence, completeness, and the
composition of its members — and identify issues that need attention.

You have tools to investigate individual contigs and bins. Use them to
build evidence before making recommendations.

Respond with JSON only. No commentary outside the JSON."""


def round1_prompt(community: dict, uneasy: list[dict]) -> str:
    """Prompt for Round 1: Community Health Check."""
    uneasy_table = ""
    if uneasy:
        uneasy_table = "\nLow-fit members (negative fit score):\n"
        for u in uneasy:
            uneasy_table += (
                f"  {u['contig']}: fit_score={u['valence']:+.3f}, "
                f"size={u['size']:,}bp, GC={u['gc']:.3f} "
                f"(bin mean {u['community_mean_gc']:.3f}), "
                f"binners={u['n_binners']}/5\n"
            )

    return f"""Examine bin {community['name']} [{community['quality_tier']}].

Bin state:
  Members: {community.get('n_members', len(community.get('members', [])))}
  Total size: {community['total_size']:,} bp
  Wholeness: {community['completeness']:.1f}% complete
  Redundancy: {community['redundancy']:.1f}%
  TNF coherence: {community['tnf_coherence']:.4f}
  Coverage correlation: {community['coverage_correlation']:.4f}
  Graph connectivity: {community['graph_connectivity']:.4f}
  Mean fit: {community['mean_fit']:+.3f}
  Min fit: {community['min_fit']:+.3f}
  Uneasy members: {community['n_uneasy']}
{uneasy_table}
Use your tools to investigate any uneasy members and understand why they're
uncomfortable. Check their identity, graph connections, and binner assignments.

Then respond with this JSON structure:
{{
  "bin": "{community['name']}",
  "assessment": "brief narrative of bin health",
  "release": [
    {{
      "contig": "name",
      "reason": "why this contig should leave"
    }}
  ],
  "recruit": [
    {{
      "contig": "name",
      "reason": "why this unbinned contig should join"
    }}
  ],
  "concerns": ["any other observations"]
}}"""


ROUND2_SYSTEM = """You evaluate unbinned contigs seeking bin placement.

Each contig has been recognized by at least some binning algorithms
but was not placed in the consensus. You investigate each contig's identity,
connections, and fit with nearby bins to recommend placement.

You have tools to examine contigs, bins, and their relationships.
Use them to build evidence. Do not guess — investigate.

Prioritize contigs whose marker genes would increase a bin's
completeness. But never force a contig into a bin where its composition
clashes — that would harm the bin's coherence.

Respond with JSON only."""


def round2_prompt(contigs: list[dict], resonance: dict[str, list[dict]]) -> str:
    """Prompt for Round 2: Unbinned contigs seek placement."""
    contig_summaries = []
    for c in contigs:
        res = resonance.get(c["name"], [])
        top_fits = ""
        if res:
            top3 = res[:3]
            top_fits = "; ".join(
                f"{r.get('community', r.get('contig', '?'))} (score={r['score']:.3f})"
                for r in top3
            )

        contig_summaries.append(
            f"  {c['name']}: {c['size']:,}bp, GC={c['gc']:.3f}, "
            f"binners={c['n_binners']}/5, "
            f"connections={len(c.get('connections', []))}, "
            f"fit_scores=[{top_fits}]"
        )

    contig_list = "\n".join(contig_summaries)

    return f"""You evaluate {len(contigs)} unbinned contigs seeking bin placement.

Each has been assessed by the binning algorithms and measured against nearby bins.
Use your tools to investigate their identity, connections, and fit scores
before making recommendations.

Contigs:
{contig_list}

For each contig, investigate using tools and then recommend:
- "join" — if there's a clear fit with a bin
- "wait" — if signals conflict (needs further investigation)
- "wander" — if no bin fits (may seed a new one)

Respond with this JSON structure:
{{
  "decisions": [
    {{
      "contig": "name",
      "action": "join|wait|wander",
      "bin": "bin_name or null",
      "evidence": "brief summary of investigation",
      "fit_score": 0.0
    }}
  ]
}}"""


ROUND3_SYSTEM = """You evaluate candidate new bins formed from unbinned contigs.

These are contigs that no binner recognized and no graph connects to existing
bins. They have been clustered by composition and abundance. You
evaluate whether each cluster represents a real organism or noise.

Signs of a real genome:
- High composition similarity (>0.9 TNF cosine coherence)
- Correlated coverage across samples
- Consistent ancestry
- Presence of essential life-function genes
- Reasonable genome size for the taxonomic group

Respond with JSON only."""


def round3_prompt(clusters: list[dict]) -> str:
    """Prompt for Round 3: Unbinned clusters."""
    cluster_summaries = []
    for cl in clusters:
        cluster_summaries.append(
            f"  Cluster {cl['id']}: {cl['n_members']} contigs, "
            f"{cl['total_size']:,}bp, "
            f"TNF coherence={cl['tnf_coherence']:.3f}, "
            f"Coverage correlation={cl.get('coverage_correlation', 0):.3f}, "
            f"Mean GC={cl.get('mean_gc', 0):.3f}"
        )

    cluster_list = "\n".join(cluster_summaries)

    return f"""{len(clusters)} clusters have emerged from the unbinned contigs —
fragments that no binner recognized and no graph connects to existing bins.

Candidate bins:
{cluster_list}

Evaluate each: does it look like a real organism or noise?

Respond with this JSON structure:
{{
  "evaluations": [
    {{
      "cluster_id": 0,
      "verdict": "accept|reject|uncertain",
      "reason": "brief assessment",
      "suggested_name": "descriptive name if accepted"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response, tolerant of markdown fences and mixed text.

    Handles cases where the model returns JSON inside markdown code blocks,
    or returns a mix of prose and JSON. Finds the outermost { ... } block.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    # Try 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try 2: strip markdown code fences
    if "```" in text:
        import re
        # Find JSON inside code fences
        fence_match = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

    # Try 3: find the outermost { ... } block
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find the matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i+1])
                    except json.JSONDecodeError:
                        pass
                    break

    raise ValueError(f"No valid JSON found in response: {text[:200]}")


# ---------------------------------------------------------------------------
# Batch grouping
# ---------------------------------------------------------------------------

def batch_contigs(contigs: list[dict], batch_size: int = 40) -> list[list[dict]]:
    """Group contigs into batches for efficient API calls.

    Tries to keep contigs with shared graph neighbors together.
    """
    # Simple size-based batching for now
    batches = []
    for i in range(0, len(contigs), batch_size):
        batches.append(contigs[i:i + batch_size])
    return batches
