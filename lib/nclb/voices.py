"""Voices — LLM prompt templates and contig-agent tool definitions.

The contigs are represented by LLM agents. Each conversation round has prompt
templates and tool definitions that let the LLM investigate on behalf
of contigs and communities.

Model hierarchy (the wisdom of scale):
  Haiku   — contig agents (Round 2: unbinned contigs, Round 3: unrecognized clusters)
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
        self._cache: dict[tuple, dict] = {}
        # Detect total number of binners from data
        self.total_binners = 0
        for c in identities.values():
            if c.binner_assignments:
                self.total_binners = len(c.binner_assignments)
                break
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

    @staticmethod
    def _prune(d: dict) -> dict:
        """Remove keys whose values are None, False, 0, or empty list/str."""
        return {k: v for k, v in d.items()
                if v is not None and v is not False and v != 0 and v != [] and v != ""}

    def get_contig_info(self, contig_name: str) -> dict:
        """Full identity card for a contig."""
        c = self.identities.get(contig_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        d: dict = {
            "name": c.name,
            "len": c.size,
            "gc": round(c.gc, 4),
            "cov": round(c.assembly_coverage, 2),
            "n_binners": f"{c.n_binners}/{self.total_binners}",
            "bin": self._display(c.community),
            "ancestry": c.ancestry,
            "n_cds": c.n_cds,
        }
        # Only include if true/non-default
        if c.is_circular:
            d["circular"] = True
        if c.is_repeat:
            d["repeat"] = True
            d["multiplicity"] = c.multiplicity
        if c.gene_names:
            d["genes"] = c.gene_names
        if c.marker_genes:
            d["scgs"] = c.marker_genes
        if len(c.connections) > 0:
            d["n_neighbors"] = len(c.connections)
            read_linked = sum(1 for r in c.connections.values() if r > 0)
            if read_linked > 0:
                d["read_linked_neighbors"] = read_linked
        # UMAP
        if c.landscape_x != 0.0 or c.landscape_y != 0.0:
            d["umap_pos"] = [round(c.landscape_x, 2), round(c.landscape_y, 2)]
            if c.landscape_cluster >= 0:
                d["umap_cl"] = c.landscape_cluster
        # MGE — only include positive findings
        if c.mge_type:
            d["mge"] = c.mge_type
        if c.is_viral:
            d["viral"] = {"score": round(c.virus_score, 4)}
            if c.virus_hallmarks:
                d["viral"]["hallmarks"] = c.virus_hallmarks
            if c.virus_taxonomy:
                d["viral"]["tax"] = c.virus_taxonomy
            if c.checkv_quality:
                d["viral"]["checkv"] = c.checkv_quality
            if c.viral_genes:
                d["viral"]["v_genes"] = c.viral_genes
            if c.host_genes:
                d["viral"]["h_genes"] = c.host_genes
        if c.is_plasmid:
            d["plasmid"] = {"score": round(c.plasmid_score, 4)}
            if c.plasmid_hallmarks:
                d["plasmid"]["hallmarks"] = c.plasmid_hallmarks
            if c.conjugation_genes:
                d["plasmid"]["conj"] = c.conjugation_genes
            if c.amr_genes:
                d["plasmid"]["amr"] = c.amr_genes
        if c.is_provirus:
            d["provirus"] = {"len": c.proviral_length}
        # HGT / defense — only include positive findings
        if c.has_integron:
            d["integrons"] = c.integrons
        if c.has_genomic_island:
            d["islands"] = c.genomic_islands
        if c.has_secretion_system:
            d["secretion"] = c.secretion_systems
        if c.has_defense_system:
            d["defense"] = c.defense_systems
        if c.coding_density:
            d["coding_density"] = round(c.coding_density, 4)
        # Domain — only if classified
        if c.domain_class:
            d["domain"] = c.domain_class
        if c.domain_confidence:
            d["domain_conf"] = c.domain_confidence
        if c.organellar_subtype:
            d["organellar"] = c.organellar_subtype
        return self._prune(d)

    def get_graph_neighbors(self, contig_name: str) -> dict:
        """Assembly graph neighbors with their bin assignments and read support."""
        c = self.identities.get(contig_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        neighbors = []
        for n, reads in c.connections.items():
            nc = self.identities.get(n)
            if nc:
                entry = {
                    "name": n,
                    "len": nc.size,
                    "gc": round(nc.gc, 4),
                }
                if reads > 0:
                    entry["reads"] = reads
                neighbors.append(entry)
        return {"contig": contig_name, "neighbors": neighbors}

    def get_binner_assignments(self, contig_name: str) -> dict:
        """All 5 binner assignments for a contig."""
        c = self.identities.get(contig_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        return {
            "contig": contig_name,
            "binner_assignments": c.binner_assignments,
            "n_binners": f"{c.n_binners}/{self.total_binners}",
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

        # Coverage correlation — None when insufficient data
        cov_r = None
        if comm.mean_coverage is not None and c.coverage is not None:
            if len(c.coverage) > 1 and np.std(c.coverage) > 0 and np.std(comm.mean_coverage) > 0:
                r, _ = pearsonr(c.coverage, comm.mean_coverage)
                cov_r = max(0.0, r)
            elif len(c.coverage) == 1 and comm.mean_coverage[0] > 0:
                ratio = c.coverage[0] / comm.mean_coverage[0]
                cov_r = 1.0 - min(abs(np.log2(max(ratio, 0.01))), 3.0) / 3.0
                cov_r = max(0.0, cov_r)

        member_set = set(comm.members)
        neighbors_in = [n for n in c.connections if n in member_set]
        has_graph = len(c.connections) > 0
        neighbor_frac = len(neighbors_in) / len(c.connections) if has_graph else None

        # Track which signals contributed to fit_score
        available_signals = ["tnf"]
        if cov_r is not None:
            available_signals.append("coverage")
        if has_graph:
            available_signals.append("graph")

        result: dict = {
            "contig": contig_name,
            "bin": self._display(bin_name),
            "fit": round(v, 4),
            "tnf_cos": round(tnf_cos, 4),
            "tnf_z": round(abs(tnf_cos - comm.tnf_coherence) / comm.tnf_sim_stdev, 2) if comm.tnf_sim_stdev > 0 else 0.0,
            "gc_z": round(abs(c.gc - comm.mean_gc) / comm.gc_stdev, 2) if comm.gc_stdev > 0 else 0.0,
        }
        if cov_r is not None:
            result["cov_r"] = round(cov_r, 4)
        if neighbor_frac is not None:
            result["graph_frac"] = round(neighbor_frac, 4)
            if neighbors_in:
                result["graph_in"] = neighbors_in
                reads_in = sum(c.connections.get(n, 0) for n in neighbors_in)
                if reads_in > 0:
                    result["reads_in"] = reads_in
        return result

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
            "len": comm.total_size,
            "gc": round(comm.mean_gc, 4),
            "gc_sd": round(comm.gc_stdev, 4),
            "complete": round(comm.completeness, 2),
            "redundancy": round(comm.redundancy, 2),
            "tier": comm.quality_tier,
            "tnf_coh": round(comm.tnf_coherence, 4),
            "cov_cor": round(comm.coverage_correlation, 4),
            "graph_conn": round(comm.graph_connectivity, 4),
            "n_missing_scg": len(comm.missing_markers),
        }
        if comm.completeness == 0 and comm.total_size < 500000:
            d["note"] = "below SCG detection threshold"
        return self._prune(d)

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
            "len_delta": c.size,
            "new_total_len": new_size,
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
        """Which bins is a contig graph-connected to (with read support)?"""
        bridges = find_graph_bridges(contig_name, self.communities, self.adjacency)
        return {
            "contig": contig_name,
            "bin_connections": [
                {"bin": self._display(name), "edges": info["edges"], "reads": info["reads"]}
                for name, info in sorted(bridges.items(), key=lambda x: -x[1]["reads"])
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

    def get_taxonomy(self, contig_name: str) -> dict:
        """All taxonomy sources for a contig."""
        c = self.identities.get(contig_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        result = {"contig": contig_name, "sources": {}}
        for source, data in c.taxonomy.items():
            result["sources"][source] = data
        result["primary_lineage"] = c.ancestry
        result["n_sources_classified"] = len(c.taxonomy)
        # Agreement check: do all sources agree at genus level?
        genera = set()
        for data in c.taxonomy.values():
            lineage = data.get("lineage", "")
            parts = [p.strip() for p in lineage.split(";")]
            # GTDB-style prefixed lineage (sendsketch, kraken2)
            prefixed = [p for p in parts if p.startswith("g__")]
            if prefixed:
                genera.add(prefixed[0].lower())
            elif len(parts) >= 6:
                # Unprefixed lineage (kaiju): d;p;c;o;f;g;s — genus is index 5
                genus = parts[5].strip()
                if genus:
                    genera.add(f"g__{genus}".lower())
        result["genus_agreement"] = len(genera) > 0 and len(genera) <= 1
        return result

    def clear_cache(self):
        """Clear the tool result cache (call between conversations)."""
        self._cache.clear()

    def dispatch(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch a tool call from the LLM, with per-conversation caching."""
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
            "get_taxonomy": self.get_taxonomy,
        }

        fn = dispatch_map.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}

        # Cache key: (tool_name, sorted argument pairs)
        cache_key = (tool_name, tuple(sorted(arguments.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = fn(**arguments)
        self._cache[cache_key] = result
        return result


# ---------------------------------------------------------------------------
# Claude API tool definitions
# ---------------------------------------------------------------------------

CONTIG_TOOLS_ANTHROPIC = [
    {
        "name": "get_contig_info",
        "description": "Returns contig metadata: length, GC%, coverage, taxonomy, domain, gene names, marker genes, MGE status, consensus bin assignment, n_binners (how many individual tools binned it anywhere — not specific to any consensus bin).",
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
        "description": "Returns bin profile: members, total length, completeness, coverage, coherence metrics, quality tier.",
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
        "description": "Predicts the impact of adding a contig to a bin: length change, GC shift, new marker gene contributions.",
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
    {
        "name": "get_taxonomy",
        "description": "Returns taxonomy classifications from all available sources (Kaiju, SendSketch, Kraken2). Shows lineage, confidence metrics, and whether sources agree at genus level.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Contig name"},
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
