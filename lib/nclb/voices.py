"""Voices — LLM prompt templates and contig-agent tool definitions.

The contigs speak through Claude. Each conversation round has prompt
templates and tool definitions that let the LLM investigate on behalf
of contigs and communities.

Model hierarchy (the wisdom of scale):
  Haiku   — contig voices (Round 2: unhoused speak, Round 3: voiceless clusters)
  Sonnet  — Elder voices (Round 1: community health, Elder investigations)
  Opus    — Contigsattva voices (mentoring, dispute mediation, Mediator)
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr

from .identity import ContigIdentity, CommunityProfile
from .valence import contig_valence, tnf_coherence, coverage_coherence
from .graph import graph_connectivity, find_graph_bridges


# ---------------------------------------------------------------------------
# Model hierarchy: the wisdom of scale
# ---------------------------------------------------------------------------

MODEL_TIERS = {
    "contig": "claude-haiku-4-5-20251001",       # the many: fast, cheap, for 28K contigs
    "elder": "claude-sonnet-4-5-20250929",        # the wise: balanced, for community-level reasoning
    "contigsattva": "claude-opus-4-6",            # the enlightened: deep, for mentoring + mediation
    "mediator": "claude-opus-4-6",                # the mediator speaks with opus wisdom
}


def model_for_role(role: str) -> str:
    """Get the appropriate Claude model for a given role."""
    return MODEL_TIERS.get(role, MODEL_TIERS["elder"])


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
    ):
        self.identities = identities
        self.communities = communities
        self.adjacency = adjacency
        self.resonance_map = resonance_map

    def who_am_i(self, contig_name: str) -> dict:
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
            "coverage_per_sample": [round(float(x), 4) for x in c.coverage],
            "graph_neighbors": c.connections,
            "n_graph_neighbors": len(c.connections),
            "testimony": c.testimony,
            "voice_strength": c.voice_strength,
            "community": c.community,
            "membership_type": c.membership_type,
            "ancestry": c.ancestry,
            "gifts": c.gifts,
            "marker_genes": c.marker_genes,
        }
        # Landscape position (UMAP coordinates + cluster centroid)
        if c.landscape_x != 0.0 or c.landscape_y != 0.0:
            d["position"] = [round(c.landscape_x, 2), round(c.landscape_y, 2)]
            if c.landscape_cluster >= 0:
                d["landscape_cluster"] = c.landscape_cluster
                d["cluster_centroid"] = [round(c.landscape_cluster_cx, 2),
                                         round(c.landscape_cluster_cy, 2)]
        # MGE annotations
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
        # Integron annotations
        if c.has_integron:
            d["has_integron"] = True
            d["integrons"] = c.integrons
        # Genomic island annotations
        if c.has_genomic_island:
            d["has_genomic_island"] = True
            d["genomic_islands"] = c.genomic_islands
        # Secretion / conjugation system annotations
        if c.has_secretion_system:
            d["has_secretion_system"] = True
            d["secretion_systems"] = c.secretion_systems
        # Defense system annotations
        if c.has_defense_system:
            d["has_defense_system"] = True
            d["defense_systems"] = c.defense_systems
        return d

    def who_are_my_neighbors(self, contig_name: str) -> dict:
        """Assembly graph neighbors with their community status."""
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
                    "community": nc.community,
                    "gc": round(nc.gc, 4),
                })
        return {"contig": contig_name, "neighbors": neighbors}

    def what_did_the_oracles_say(self, contig_name: str) -> dict:
        """All 5 binner assignments for a contig."""
        c = self.identities.get(contig_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        return {
            "contig": contig_name,
            "testimony": c.testimony,
            "voice_strength": c.voice_strength,
            "consensus": c.community,
        }

    def how_do_i_resonate_with(self, contig_name: str, community_name: str) -> dict:
        """Compute valence of a contig against a specific community.

        Returns both computed metrics AND raw data for the LLM to assess.
        """
        c = self.identities.get(contig_name)
        comm = self.communities.get(community_name)
        if not c:
            return {"error": f"Unknown contig: {contig_name}"}
        if not comm:
            return {"error": f"Unknown community: {community_name}"}

        v = contig_valence(c, comm, self.adjacency)

        # Individual signal components
        harmony = 0.0
        if comm.tnf_centroid is not None and c.tnf is not None:
            harmony = 1.0 - cosine_distance(c.tnf, comm.tnf_centroid)

        rhythm = 0.0
        if comm.mean_coverage is not None and c.coverage is not None:
            if len(c.coverage) > 1 and np.std(c.coverage) > 0 and np.std(comm.mean_coverage) > 0:
                r, _ = pearsonr(c.coverage, comm.mean_coverage)
                rhythm = max(0.0, r)

        member_set = set(comm.members)
        neighbors_in = [n for n in c.connections if n in member_set]
        kinship = len(neighbors_in) / len(c.connections) if c.connections else 0.0

        return {
            "contig": contig_name,
            "community": community_name,
            "valence": round(v, 4),
            "harmony_tnf_cosine": round(harmony, 4),
            "rhythm_coverage_correlation": round(rhythm, 4),
            "kinship_fraction": round(kinship, 4),
            "graph_neighbors_in_community": neighbors_in,
            "n_graph_neighbors_in": len(neighbors_in),
            "contig_coverage": [round(float(x), 4) for x in c.coverage],
            "community_mean_coverage": [round(float(x), 4) for x in comm.mean_coverage] if comm.mean_coverage is not None else [],
            "contig_gc": round(c.gc, 4),
            "community_mean_gc": round(comm.mean_gc, 4),
            "gc_delta": round(abs(c.gc - comm.mean_gc), 4),
            "community_completeness": round(comm.completeness, 2),
            "community_elder_rank": comm.elder_rank,
        }

    def what_is_this_community(self, community_name: str) -> dict:
        """Full community profile."""
        comm = self.communities.get(community_name)
        if not comm:
            return {"error": f"Unknown community: {community_name}"}
        d = {
            "name": comm.name,
            "source_binner": comm.source_binner,
            "n_members": len(comm.members),
            "total_size": comm.total_size,
            "n50": comm.n50,
            "mean_gc": round(comm.mean_gc, 4),
            "gc_stdev": round(comm.gc_stdev, 4),
            "completeness": round(comm.completeness, 2),
            "redundancy": round(comm.redundancy, 2),
            "elder_rank": comm.elder_rank,
            "tnf_coherence": round(comm.tnf_coherence, 4),
            "coverage_correlation": round(comm.coverage_correlation, 4),
            "graph_connectivity": round(comm.graph_connectivity, 4),
            "missing_markers": comm.missing_markers,
        }
        # Include mean coverage profile (actual data)
        if comm.mean_coverage is not None:
            d["mean_coverage_per_sample"] = [round(float(x), 4) for x in comm.mean_coverage]
        return d

    def what_gifts_are_missing(self, community_name: str) -> dict:
        """Marker genes the community still needs."""
        comm = self.communities.get(community_name)
        if not comm:
            return {"error": f"Unknown community: {community_name}"}
        return {
            "community": community_name,
            "completeness": round(comm.completeness, 2),
            "missing_markers": comm.missing_markers,
            "n_missing": len(comm.missing_markers),
        }

    def what_would_change_if_i_joined(self, contig_name: str, community_name: str) -> dict:
        """Predict impact of a contig joining a community."""
        c = self.identities.get(contig_name)
        comm = self.communities.get(community_name)
        if not c or not comm:
            return {"error": "Unknown contig or community"}

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

        # Would any existing member's valence drop?
        # (approximate: check if new centroid shifts significantly)
        gc_shift = abs(new_mean_gc - old_mean_gc)

        return {
            "contig": contig_name,
            "community": community_name,
            "size_delta": c.size,
            "new_total_size": new_size,
            "gc_shift": round(gc_shift, 4),
            "contributed_markers": contributed_markers,
            "n_contributed": len(contributed_markers),
        }

    def who_resonates_with_me(self, contig_name: str, k: int = 10) -> dict:
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
                    "community": self.identities[name].community if name in self.identities else None,
                }
                for name, sim in neighbors
            ],
        }

    def find_graph_connections(self, contig_name: str) -> dict:
        """Which communities is a contig graph-connected to?"""
        bridges = find_graph_bridges(contig_name, self.communities, self.adjacency)
        return {
            "contig": contig_name,
            "community_connections": [
                {"community": name, "n_edges": n}
                for name, n in sorted(bridges.items(), key=lambda x: -x[1])
            ],
        }

    def dispatch(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch a tool call from the LLM."""
        dispatch_map = {
            "who_am_i": self.who_am_i,
            "who_are_my_neighbors": self.who_are_my_neighbors,
            "what_did_the_oracles_say": self.what_did_the_oracles_say,
            "how_do_i_resonate_with": self.how_do_i_resonate_with,
            "what_is_this_community": self.what_is_this_community,
            "what_gifts_are_missing": self.what_gifts_are_missing,
            "what_would_change_if_i_joined": self.what_would_change_if_i_joined,
            "who_resonates_with_me": self.who_resonates_with_me,
            "find_graph_connections": self.find_graph_connections,
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
        "name": "who_am_i",
        "description": "Returns the full identity card for a contig — composition, energy, ancestry, gifts, connections, and oracle testimony.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
            },
            "required": ["contig_name"],
        },
    },
    {
        "name": "who_are_my_neighbors",
        "description": "Returns assembly graph neighbors of a contig with their community status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
            },
            "required": ["contig_name"],
        },
    },
    {
        "name": "how_do_i_resonate_with",
        "description": "Computes how well a contig resonates with a specific community. Returns valence, harmony (TNF cosine), rhythm (coverage correlation), kinship (graph), raw coverage profiles, and GC comparison.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
                "community_name": {"type": "string", "description": "Name of the community"},
            },
            "required": ["contig_name", "community_name"],
        },
    },
    {
        "name": "what_is_this_community",
        "description": "Returns the full profile of a community — members, size, completeness, coverage profile, harmony metrics, and elder rank.",
        "input_schema": {
            "type": "object",
            "properties": {
                "community_name": {"type": "string", "description": "Name of the community"},
            },
            "required": ["community_name"],
        },
    },
    {
        "name": "what_gifts_are_missing",
        "description": "Lists marker genes that a community still needs for completeness.",
        "input_schema": {
            "type": "object",
            "properties": {
                "community_name": {"type": "string", "description": "Name of the community"},
            },
            "required": ["community_name"],
        },
    },
    {
        "name": "what_would_change_if_i_joined",
        "description": "Predicts the impact of a contig joining a community — size change, GC shift, marker gene contribution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
                "community_name": {"type": "string", "description": "Name of the community"},
            },
            "required": ["contig_name", "community_name"],
        },
    },
    {
        "name": "find_graph_connections",
        "description": "Finds which communities a contig is connected to via the assembly graph.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Name of the contig"},
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

ROUND1_SYSTEM = """You are the voice of metagenome-assembled genome communities.

Each community is a group of contigs that were placed together by consensus binning.
You examine each community's collective state — its harmony, wholeness, and the
wellbeing of its members — and identify issues that need attention.

You have tools to investigate individual contigs and communities. Use them to
build evidence before making recommendations.

Respond with JSON only. No commentary outside the JSON."""


def round1_prompt(community: dict, uneasy: list[dict]) -> str:
    """Prompt for Round 1: Community Health Check."""
    uneasy_table = ""
    if uneasy:
        uneasy_table = "\nUneasy members (negative valence):\n"
        for u in uneasy:
            uneasy_table += (
                f"  {u['contig']}: valence={u['valence']:+.3f}, "
                f"size={u['size']:,}bp, GC={u['gc']:.3f} "
                f"(community mean {u['community_mean_gc']:.3f}), "
                f"voice={u['voice_strength']}/5\n"
            )

    return f"""Examine community {community['name']} [{community['elder_rank']}].

Community state:
  Members: {community.get('n_members', len(community.get('members', [])))}
  Total size: {community['total_size']:,} bp
  Wholeness: {community['completeness']:.1f}% complete
  Redundancy: {community['redundancy']:.1f}%
  TNF coherence: {community['tnf_coherence']:.4f}
  Coverage correlation: {community['coverage_correlation']:.4f}
  Graph connectivity: {community['graph_connectivity']:.4f}
  Mean valence: {community['mean_valence']:+.3f}
  Min valence: {community['min_valence']:+.3f}
  Uneasy members: {community['n_uneasy']}
{uneasy_table}
Use your tools to investigate any uneasy members and understand why they're
uncomfortable. Check their identity, graph connections, and oracle testimony.

Then respond with this JSON structure:
{{
  "community": "{community['name']}",
  "assessment": "brief narrative of community health",
  "release": [
    {{
      "contig": "name",
      "reason": "why this contig should leave"
    }}
  ],
  "recruit": [
    {{
      "contig": "name",
      "reason": "why this unhoused contig should join"
    }}
  ],
  "concerns": ["any other observations"]
}}"""


ROUND2_SYSTEM = """You speak for unhoused contigs seeking community.

Each contig has been recognized by at least some oracles (binning algorithms)
but was not placed in the consensus. You investigate each contig's identity,
connections, and resonance with nearby communities to recommend placement.

You have tools to examine contigs, communities, and their relationships.
Use them to build evidence. Do not guess — investigate.

Prioritize contigs whose gifts (marker genes) would increase a community's
wholeness. But never force a contig into a community where its composition
clashes — that would harm the community's harmony.

Respond with JSON only."""


def round2_prompt(contigs: list[dict], resonance: dict[str, list[dict]]) -> str:
    """Prompt for Round 2: Unhoused contigs speak."""
    contig_summaries = []
    for c in contigs:
        res = resonance.get(c["name"], [])
        top_resonance = ""
        if res:
            top3 = res[:3]
            top_resonance = "; ".join(
                f"{r.get('community', r.get('contig', '?'))} (score={r['score']:.3f})"
                for r in top3
            )

        contig_summaries.append(
            f"  {c['name']}: {c['size']:,}bp, GC={c['gc']:.3f}, "
            f"voice={c['voice_strength']}/5, "
            f"connections={len(c.get('connections', []))}, "
            f"resonance=[{top_resonance}]"
        )

    contig_list = "\n".join(contig_summaries)

    return f"""You speak for {len(contigs)} unhoused contigs seeking community.

Each has been heard by the oracles and measured against nearby communities.
Use your tools to investigate their identity, connections, and resonance
before making recommendations.

Contigs:
{contig_list}

For each contig, investigate using tools and then recommend:
- "join" — if there's clear resonance with a community
- "wait" — if signals conflict (needs further investigation)
- "wander" — if no community resonates (may seed a new one)

Respond with this JSON structure:
{{
  "decisions": [
    {{
      "contig": "name",
      "action": "join|wait|wander",
      "community": "community_name or null",
      "evidence": "brief summary of investigation",
      "valence": 0.0
    }}
  ]
}}"""


ROUND3_SYSTEM = """You evaluate candidate new communities formed from voiceless contigs.

These are contigs that no binner recognized and no graph connects to existing
communities. They have been clustered by composition and abundance. You
evaluate whether each cluster represents a real organism or noise.

Signs of a real community:
- High composition harmony (>0.9 cosine coherence)
- Synchronized energy across samples
- Consistent ancestry
- Presence of essential life-function genes
- Reasonable genome size for the taxonomic group

Respond with JSON only."""


def round3_prompt(clusters: list[dict]) -> str:
    """Prompt for Round 3: Voiceless clusters."""
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

    return f"""{len(clusters)} clusters have emerged from the voiceless contigs —
fragments that no oracle recognized and no graph connects to existing communities.

Candidate communities:
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
