"""Mediator — conflict resolution across all proposals.

After all three rounds of conversation, proposals may conflict:
- Two bins want the same contig
- A released contig may match a new bin
- New bin clusters may overlap with existing ones

The Mediator sees all proposals and resolves conflicts with a single
LLM call, then applies the results deterministically.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from .identity import ContigIdentity, CommunityProfile
from .valence import contig_fit_score


# ---------------------------------------------------------------------------
# Proposal extraction
# ---------------------------------------------------------------------------

def extract_proposals(all_proposals: list[dict]) -> dict:
    """Extract actionable proposals from all three rounds.

    Returns:
        {
            "releases": [(contig, from_community, reason)],
            "joins": [(contig, to_community, evidence, valence)],
            "waits": [(contig, evidence)],
            "wanders": [(contig, evidence)],
            "new_communities": [(cluster_id, members, name, reason)],
            "recruits": [(contig, to_community, reason)],
        }
    """
    releases = []
    joins = []
    waits = []
    wanders = []
    new_communities = []
    recruits = []

    for proposal in all_proposals:
        round_num = proposal.get("round", 0)
        result = proposal.get("result", {})

        if round_num == 1:
            bin_name = proposal.get("bin", "")
            for r in result.get("release", []):
                releases.append((r["contig"], bin_name, r.get("reason", "")))
            for r in result.get("recruit", []):
                recruits.append((r["contig"], bin_name, r.get("reason", "")))

        elif round_num == 2:
            for d in result.get("decisions", []):
                action = d.get("action", "wait")
                if action == "join" and d.get("bin"):
                    joins.append((
                        d["contig"],
                        d["bin"],
                        d.get("evidence", ""),
                        d.get("fit_score", 0.0),
                    ))
                elif action == "wait":
                    waits.append((d["contig"], d.get("evidence", "")))
                elif action == "wander":
                    wanders.append((d["contig"], d.get("evidence", "")))

        elif round_num == 3:
            clusters = proposal.get("clusters", [])
            for ev in result.get("evaluations", []):
                if ev.get("verdict") == "accept":
                    cluster_id = ev["cluster_id"]
                    matching = [c for c in clusters if c["id"] == cluster_id]
                    if matching:
                        new_communities.append((
                            cluster_id,
                            matching[0]["member_names"],
                            ev.get("suggested_name", f"new_bin_{cluster_id}"),
                            ev.get("reason", ""),
                        ))

    return {
        "releases": releases,
        "joins": joins,
        "waits": waits,
        "wanders": wanders,
        "new_communities": new_communities,
        "recruits": recruits,
    }


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def detect_conflicts(proposals: dict) -> list[dict]:
    """Find conflicts in the proposals."""
    conflicts = []

    # Contigs claimed by multiple communities
    contig_claims: dict[str, list[tuple[str, float]]] = {}
    for contig, community, evidence, valence in proposals["joins"]:
        contig_claims.setdefault(contig, []).append((community, valence))
    for contig, community, reason in proposals["recruits"]:
        contig_claims.setdefault(contig, []).append((community, 0.0))

    for contig, claims in contig_claims.items():
        if len(claims) > 1:
            conflicts.append({
                "type": "multi_claim",
                "contig": contig,
                "claims": [{"bin": c, "fit_score": v} for c, v in claims],
            })

    # Contigs both released and recruited
    released_contigs = {r[0] for r in proposals["releases"]}
    for contig, community, evidence, valence in proposals["joins"]:
        if contig in released_contigs:
            conflicts.append({
                "type": "release_rejoin",
                "contig": contig,
                "released_from": next(
                    r[1] for r in proposals["releases"] if r[0] == contig
                ),
                "joining": community,
            })

    return conflicts


# ---------------------------------------------------------------------------
# Deterministic resolution (no LLM needed for simple cases)
# ---------------------------------------------------------------------------

def resolve_deterministic(
    proposals: dict,
    identities: dict[str, ContigIdentity],
    communities: dict[str, CommunityProfile],
    adjacency: dict[str, list[str]],
) -> dict:
    """Resolve proposals deterministically where possible.

    For multi-claim conflicts, pick the community with highest fit score.
    For release+rejoin, allow the rejoin if fit score is positive.

    Returns resolved proposals in the same structure.
    """
    conflicts = detect_conflicts(proposals)

    # Build a set of contigs with conflicts for special handling
    conflict_contigs = set()
    for c in conflicts:
        conflict_contigs.add(c["contig"])

    resolved_joins = []
    resolved_releases = list(proposals["releases"])
    resolved_recruits = []

    # Handle non-conflicted joins directly
    for contig, community, evidence, valence in proposals["joins"]:
        if contig not in conflict_contigs:
            resolved_joins.append((contig, community, evidence, valence))

    for contig, community, reason in proposals["recruits"]:
        if contig not in conflict_contigs:
            resolved_recruits.append((contig, community, reason))

    # Resolve multi-claim conflicts by fit score
    for conflict in conflicts:
        if conflict["type"] == "multi_claim":
            contig_name = conflict["contig"]
            contig = identities.get(contig_name)
            if not contig:
                continue

            best_community = None
            best_valence = -2.0

            for claim in conflict["claims"]:
                comm = communities.get(claim["bin"])
                if comm:
                    v = contig_fit_score(contig, comm, adjacency)
                    if v > best_valence:
                        best_valence = v
                        best_community = claim["bin"]

            if best_community and best_valence > -0.5:
                resolved_joins.append((
                    contig_name,
                    best_community,
                    f"Resolved multi-claim: highest fit score ({best_valence:+.3f})",
                    best_valence,
                ))

        elif conflict["type"] == "release_rejoin":
            contig_name = conflict["contig"]
            new_comm = conflict["joining"]
            contig = identities.get(contig_name)
            comm = communities.get(new_comm)

            if contig and comm:
                v = contig_fit_score(contig, comm, adjacency)
                if v > 0:
                    resolved_joins.append((
                        contig_name,
                        new_comm,
                        f"Release+rejoin: positive fit score ({v:+.3f}) in new bin",
                        v,
                    ))

    return {
        "releases": resolved_releases,
        "joins": resolved_joins,
        "waits": proposals["waits"],
        "wanders": proposals["wanders"],
        "new_communities": proposals["new_communities"],
        "recruits": resolved_recruits,
    }


# ---------------------------------------------------------------------------
# Mediation prompt (for complex conflicts needing LLM)
# ---------------------------------------------------------------------------

MEDIATOR_SYSTEM = """You are the Mediator of the Assembly.

You see all proposals from the three conversation rounds and resolve
any remaining conflicts. Your goal is to maximize collective coherence
while respecting individual identity.

Principles:
1. A contig goes where its fit score is highest
2. No contig should be forced into a bin where it clashes
3. If signals genuinely conflict, prefer "wait" over a bad placement
4. New bins are welcome if their members truly fit

Respond with JSON only."""


def mediator_prompt(
    proposals: dict,
    conflicts: list[dict],
    stats: dict,
) -> str:
    """Build the Mediator prompt for complex conflict resolution."""
    conflict_desc = ""
    for c in conflicts:
        if c["type"] == "multi_claim":
            claims = ", ".join(
                f"{cl['bin']} (fit_score={cl['fit_score']:+.3f})"
                for cl in c["claims"]
            )
            conflict_desc += f"\n  {c['contig']} claimed by: {claims}"
        elif c["type"] == "release_rejoin":
            conflict_desc += (
                f"\n  {c['contig']} released from {c['released_from']} "
                f"but wants to join {c['joining']}"
            )

    return f"""The three rounds of conversation have produced these proposals:

Summary:
  Releases: {len(proposals['releases'])}
  Joins: {len(proposals['joins'])}
  Waits: {len(proposals['waits'])}
  Wanders: {len(proposals['wanders'])}
  New bins: {len(proposals['new_communities'])}
  Recruits: {len(proposals['recruits'])}

Conflicts requiring resolution:{conflict_desc or " (none)"}

Assembly context:
  Total contigs: {stats.get('total_contigs', 0):,}
  Currently housed: {stats.get('housed', 0):,}
  Bins: {stats.get('total_communities', 0)}

For each conflict, decide:
- Which bin gets the contig (highest fit score wins by default)
- Whether release+rejoin should be allowed

Respond with:
{{
  "resolutions": [
    {{
      "contig": "name",
      "action": "join|release|wait",
      "bin": "target_bin or null",
      "reason": "brief explanation"
    }}
  ],
  "approved_new_bins": [cluster_id, ...],
  "commentary": "overall assessment"
}}"""


# ---------------------------------------------------------------------------
# Apply resolved proposals
# ---------------------------------------------------------------------------

def apply_proposals(
    resolved: dict,
    identities: dict[str, ContigIdentity],
    communities: dict[str, CommunityProfile],
) -> dict:
    """Apply resolved proposals to update identities and communities.

    Returns a summary of changes made.
    """
    changes = {
        "released": [],
        "joined": [],
        "new_communities_founded": [],
    }

    # Step 1: Releases
    for contig_name, from_community, reason in resolved["releases"]:
        c = identities.get(contig_name)
        if c and c.community == from_community:
            c.community = None
            c.membership_type = "unbinned"
            c.fit_score = -1.0

            comm = communities.get(from_community)
            if comm and contig_name in comm.members:
                comm.members.remove(contig_name)

            changes["released"].append({
                "contig": contig_name,
                "from": from_community,
                "reason": reason,
            })

    # Step 2: Joins (from Round 2 + resolved recruits)
    all_joins = list(resolved["joins"])
    for contig_name, community, reason in resolved["recruits"]:
        all_joins.append((contig_name, community, reason, 0.0))

    for contig_name, community, evidence, valence in all_joins:
        c = identities.get(contig_name)
        comm = communities.get(community)
        if c and comm and c.community is None:
            c.community = community
            c.membership_type = "core"
            c.fit_score = valence

            if contig_name not in comm.members:
                comm.members.append(contig_name)

            changes["joined"].append({
                "contig": contig_name,
                "to": community,
                "evidence": evidence,
                "fit_score": valence,
            })

    # Step 3: New communities
    for cluster_id, members, name, reason in resolved["new_communities"]:
        new_comm = CommunityProfile(
            name=name,
            source_binner="nclb",
            members=members,
            quality_tier="low",
        )

        # Compute basic stats
        member_sizes = []
        member_tnf = []
        member_cov = []
        for m in members:
            c = identities.get(m)
            if c:
                c.community = name
                c.membership_type = "core"
                member_sizes.append(c.size)
                member_tnf.append(c.tnf)
                member_cov.append(c.coverage)

        new_comm.total_size = sum(member_sizes)
        if member_tnf:
            new_comm.tnf_centroid = np.mean(member_tnf, axis=0)
            new_comm.mean_coverage = np.mean(member_cov, axis=0)
            new_comm.mean_gc = float(np.mean([
                identities[m].gc for m in members if m in identities
            ]))

        communities[name] = new_comm

        changes["new_communities_founded"].append({
            "name": name,
            "n_members": len(members),
            "total_size": new_comm.total_size,
            "reason": reason,
        })

    return changes
