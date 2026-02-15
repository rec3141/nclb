"""Journal — the Chronicle of the Assembly.

Generates narrative accounts of the NCLB process in both JSON and
markdown. Every decision is documented: why a contig was placed,
why a bin was founded, what an Elder discovered.

The chronicle is not just documentation — it is a new form of
scientific output that a microbiologist can read, audit, and learn from.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .identity import ContigIdentity, CommunityProfile


# ---------------------------------------------------------------------------
# Chronicle data structures
# ---------------------------------------------------------------------------

class Chronicle:
    """The story of the assembly, told from the contigs' perspective."""

    def __init__(self):
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.assembly_stats: dict = {}
        self.community_stories: list[CommunityStory] = []
        self.unhoused_stories: list[dict] = []
        self.new_community_stories: list[dict] = []
        self.elder_reports: list[dict] = []
        self.movable_stories: list[dict] = []
        self.summary: dict = {}

    def to_json(self) -> dict:
        return {
            "version": "0.1.0",
            "timestamp": self.timestamp,
            "assembly_stats": self.assembly_stats,
            "community_stories": [s.to_json() for s in self.community_stories],
            "unhoused_stories": self.unhoused_stories,
            "new_community_stories": self.new_community_stories,
            "elder_reports": self.elder_reports,
            "movable_stories": self.movable_stories,
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        lines = []
        lines.append("# No Contig Left Behind: Chronicle of the Assembly")
        lines.append("")
        lines.append(f"*Generated {self.timestamp}*")
        lines.append("")

        # Assembly overview
        stats = self.assembly_stats
        lines.append("## Assembly Overview")
        lines.append("")
        lines.append(f"- **Total contigs**: {stats.get('total_contigs', 0):,}")
        lines.append(f"- **Total size**: {stats.get('total_assembly_size', 0):,} bp")
        lines.append(f"- **Bins**: {stats.get('total_communities', 0)}")
        lines.append(f"- **Housed before NCLB**: {stats.get('housed_before', 0):,} "
                      f"({stats.get('housed_before_pct', 0):.1f}%)")
        lines.append(f"- **Housed after NCLB**: {stats.get('housed_after', 0):,} "
                      f"({stats.get('housed_after_pct', 0):.1f}%)")
        lines.append("")

        if stats.get("quality_tiers"):
            lines.append("### Quality Tiers")
            lines.append("")
            for rank, count in stats["quality_tiers"].items():
                lines.append(f"- **{rank}**: {count}")
            lines.append("")

        # Community stories
        if self.community_stories:
            lines.append("---")
            lines.append("")
            lines.append("## Bin Reports")
            lines.append("")
            for story in self.community_stories:
                lines.append(story.to_markdown())
                lines.append("")

        # Unbinned contig stories
        if self.unhoused_stories:
            lines.append("---")
            lines.append("")
            lines.append("## Contigs Who Found a Home")
            lines.append("")
            for story in self.unhoused_stories[:20]:
                lines.append(f"### {story['contig']}")
                lines.append("")
                lines.append(f"*{story['size']:,} bp, GC={story['gc']:.3f}, "
                              f"n_binners={story['n_binners']}/5*")
                lines.append("")
                if story.get("evidence"):
                    lines.append(f"> {story['evidence']}")
                    lines.append("")
                lines.append(f"Joined **{story['bin']}** "
                              f"(fit score {story.get('fit_score', 0):+.3f})")
                lines.append("")

        # New communities
        if self.new_community_stories:
            lines.append("---")
            lines.append("")
            lines.append("## New Bins Founded")
            lines.append("")
            for story in self.new_community_stories:
                lines.append(f"### {story['name']}")
                lines.append("")
                lines.append(f"*{story['n_members']} contigs, "
                              f"{story['total_size']:,} bp*")
                lines.append("")
                if story.get("reason"):
                    lines.append(f"> {story['reason']}")
                    lines.append("")

        # Elder investigations
        if self.elder_reports:
            lines.append("---")
            lines.append("")
            lines.append("## Elder Investigations")
            lines.append("")
            for report in self.elder_reports:
                lines.append(f"### {report.get('community', 'Unknown')}")
                lines.append("")
                if report.get("narrative"):
                    lines.append(report["narrative"])
                    lines.append("")
                if report.get("verdict"):
                    lines.append(f"**Verdict**: {report['verdict']}")
                    lines.append("")

        # Mobile element stories
        if self.movable_stories:
            lines.append("---")
            lines.append("")
            lines.append("## Movable Elements")
            lines.append("")
            for story in self.movable_stories:
                lines.append(f"### {story['contig']}")
                lines.append("")
                lines.append(f"*Type: {story['element_type']}, "
                              f"confidence: {story.get('confidence', 0):.0%}*")
                lines.append("")
                if story.get("communities"):
                    comms = ", ".join(story["communities"])
                    lines.append(f"Dual citizenship: {comms}")
                    lines.append("")
                if story.get("evidence"):
                    lines.append(f"> {story['evidence']}")
                    lines.append("")

        # Summary
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        summary = self.summary
        lines.append(f"- Contigs released: {summary.get('n_released', 0)}")
        lines.append(f"- Contigs placed: {summary.get('n_placed', 0)}")
        lines.append(f"- New bins: {summary.get('n_new_communities', 0)}")
        lines.append(f"- Mobile elements identified: {summary.get('n_travelers', 0)}")
        lines.append(f"- Elder investigations: {summary.get('n_elder_investigations', 0)}")
        lines.append("")

        return "\n".join(lines)


class CommunityStory:
    """Narrative for a single bin's experience during NCLB."""

    def __init__(self, community: CommunityProfile):
        self.name = community.name
        self.quality_tier = community.quality_tier
        self.completeness_before = community.completeness
        self.completeness_after = community.completeness
        self.redundancy_before = community.redundancy
        self.redundancy_after = community.redundancy
        self.members_before = len(community.members)
        self.members_after = len(community.members)
        self.released: list[dict] = []
        self.welcomed: list[dict] = []
        self.assessment = ""
        self.concerns: list[str] = []

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "quality_tier": self.quality_tier,
            "completeness_before": self.completeness_before,
            "completeness_after": self.completeness_after,
            "redundancy_before": self.redundancy_before,
            "redundancy_after": self.redundancy_after,
            "members_before": self.members_before,
            "members_after": self.members_after,
            "released": self.released,
            "welcomed": self.welcomed,
            "assessment": self.assessment,
            "concerns": self.concerns,
        }

    def to_markdown(self) -> str:
        lines = []
        lines.append(f"### {self.name} [{self.quality_tier}]")
        lines.append("")
        lines.append(f"**Wholeness**: {self.completeness_before:.1f}% → "
                      f"{self.completeness_after:.1f}%")
        lines.append(f"**Redundancy**: {self.redundancy_before:.1f}% → "
                      f"{self.redundancy_after:.1f}%")
        lines.append(f"**Members**: {self.members_before} → {self.members_after}")
        lines.append("")

        if self.assessment:
            lines.append(self.assessment)
            lines.append("")

        if self.released:
            lines.append("**Released:**")
            for r in self.released:
                lines.append(f"- {r['contig']}: {r.get('reason', '')}")
            lines.append("")

        if self.welcomed:
            lines.append("**Welcomed:**")
            for w in self.welcomed:
                lines.append(f"- {w['contig']}: {w.get('evidence', '')}")
            lines.append("")

        if self.concerns:
            lines.append("**Concerns:**")
            for c in self.concerns:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chronicle builder
# ---------------------------------------------------------------------------

def build_chronicle(
    assembly_stats: dict,
    proposals: dict,
    changes: dict,
    communities: dict[str, CommunityProfile],
    identities: dict[str, ContigIdentity],
) -> Chronicle:
    """Build the full chronicle from proposals and applied changes."""

    chronicle = Chronicle()
    chronicle.assembly_stats = assembly_stats

    # Community stories from Round 1 results + applied changes
    for name, comm in communities.items():
        story = CommunityStory(comm)

        # Check if any contigs were released from this community
        released = [c for c in changes.get("released", []) if c["from"] == name]
        story.released = released
        story.members_after = len(comm.members)

        # Check if any contigs were welcomed
        welcomed = [c for c in changes.get("joined", []) if c["to"] == name]
        story.welcomed = welcomed

        if released or welcomed:
            chronicle.community_stories.append(story)

    # Unhoused contig stories — contigs that found a home
    for joined in changes.get("joined", []):
        contig = identities.get(joined["contig"])
        if contig:
            chronicle.unhoused_stories.append({
                "contig": joined["contig"],
                "bin": joined["to"],
                "evidence": joined.get("evidence", ""),
                "fit_score": joined.get("fit_score", 0.0),
                "size": contig.size,
                "gc": contig.gc,
                "n_binners": contig.n_binners,
            })

    # New community stories
    for founded in changes.get("new_communities_founded", []):
        chronicle.new_community_stories.append(founded)

    # Summary
    chronicle.summary = {
        "n_released": len(changes.get("released", [])),
        "n_placed": len(changes.get("joined", [])),
        "n_new_communities": len(changes.get("new_communities_founded", [])),
        "n_travelers": 0,  # filled by Elder phase
        "n_elder_investigations": 0,  # filled by Elder phase
    }

    return chronicle


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_chronicle(chronicle: Chronicle, output_dir: Path):
    """Write the chronicle in both JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "chronicle.json", "w") as f:
        json.dump(chronicle.to_json(), f, indent=2)

    # Markdown
    with open(output_dir / "chronicle.md", "w") as f:
        f.write(chronicle.to_markdown())


def write_contig_membership(
    identities: dict[str, ContigIdentity],
    output_dir: Path,
):
    """Write contig membership tables.

    Produces two files:
    - contig2community.tsv: Traditional one-contig-one-bin (for tools that need it)
    - contig_membership.tsv: Full many-to-many with type + fit_score
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Traditional TSV
    with open(output_dir / "contig2community.tsv", "w") as f:
        for name, c in sorted(identities.items()):
            if c.community:
                f.write(f"{name}\t{c.community}\n")

    # Full membership table
    with open(output_dir / "contig_membership.tsv", "w") as f:
        f.write("contig\tcommunity\tmembership_type\tfit_score\tsize\tgc\tn_binners\n")
        for name, c in sorted(identities.items()):
            comm = c.community or "unbinned"
            f.write(f"{name}\t{comm}\t{c.membership_type}\t{c.fit_score:.4f}\t"
                    f"{c.size}\t{c.gc:.4f}\t{c.n_binners}\n")


def write_fit_report(
    identities: dict[str, ContigIdentity],
    output_dir: Path,
):
    """Write per-contig fit scores."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "fit_score_report.tsv", "w") as f:
        f.write("contig\tcommunity\tfit_score\tsize\tgc\tn_binners\tmembership_type\n")
        for name, c in sorted(identities.items(), key=lambda x: x[1].fit_score):
            comm = c.community or "unbinned"
            f.write(f"{name}\t{comm}\t{c.fit_score:.4f}\t{c.size}\t"
                    f"{c.gc:.4f}\t{c.n_binners}\t{c.membership_type}\n")
