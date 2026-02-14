"""Elder tools — heavyweight external investigations.

Elders are community-level agents with access to tools that individual
contigs cannot run: BLAST, phylogenetics, ANI computation, read mapping,
targeted re-assembly, and mobile element detection.

These tools wrap external bioinformatics software and are only invoked
for communities with sufficient Elder rank.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .identity import ContigIdentity, CommunityProfile


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ElderConfig:
    """Configuration for Elder tool execution."""
    blast_db: str = "nr"                      # BLAST database name
    blast_threads: int = 4
    blast_evalue: float = 1e-10
    mafft_threads: int = 4
    minimap2_threads: int = 4
    flye_threads: int = 8
    flye_genome_size: str = "500k"            # for targeted re-assembly
    assembly_fasta: Optional[Path] = None     # path to co-assembly FASTA
    reads_dir: Optional[Path] = None          # path to input reads
    work_dir: Optional[Path] = None           # scratch directory
    max_blast_hits: int = 20


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------

@dataclass
class BlastResult:
    query: str
    subject: str
    identity: float
    alignment_length: int
    evalue: float
    bitscore: float
    subject_title: str = ""
    taxonomy: str = ""


@dataclass
class PhylogenyResult:
    gene_name: str
    n_sequences: int
    newick: str                               # tree in Newick format
    interpretation: str = ""                  # filled by Elder LLM


@dataclass
class ANIResult:
    set_a_name: str
    set_b_name: str
    ani: float                                # average nucleotide identity
    n_aligned: int                            # aligned bases
    coverage: float                           # fraction of shorter set aligned


@dataclass
class MobileElementResult:
    contig_name: str
    element_type: str                         # prophage|plasmid|island|transposon|none
    confidence: float
    hallmark_genes: list[str] = field(default_factory=list)
    integration_site: str = ""
    evidence: str = ""


@dataclass
class ElderInvestigation:
    """Complete record of an Elder's investigation into SCG redundancy."""
    community_name: str
    redundant_marker: str
    contigs_with_marker: list[str]
    blast_results: list[BlastResult] = field(default_factory=list)
    phylogeny: Optional[PhylogenyResult] = None
    ani: Optional[ANIResult] = None
    verdict: str = ""                         # "ecotype"|"contamination"|"uncertain"
    narrative: str = ""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class ElderToolkit:
    """Runtime for Elder-level external tools."""

    def __init__(self, config: ElderConfig):
        self.config = config
        if config.work_dir:
            config.work_dir.mkdir(parents=True, exist_ok=True)

    def _work_dir(self) -> Path:
        if self.config.work_dir:
            return self.config.work_dir
        return Path(tempfile.mkdtemp(prefix="nclb_elder_"))

    def blast_gene(
        self,
        sequence: str,
        gene_name: str = "query",
        db: Optional[str] = None,
    ) -> list[BlastResult]:
        """BLAST a gene sequence against a database.

        Args:
            sequence: nucleotide or protein sequence
            gene_name: name for the query
            db: database to search (default: config.blast_db)
        """
        work = self._work_dir()
        query_file = work / f"{gene_name}_query.fasta"
        output_file = work / f"{gene_name}_blast.tsv"

        query_file.write_text(f">{gene_name}\n{sequence}\n")

        db = db or self.config.blast_db
        cmd = [
            "blastn",
            "-query", str(query_file),
            "-db", db,
            "-out", str(output_file),
            "-outfmt", "6 qseqid sseqid pident length evalue bitscore stitle",
            "-evalue", str(self.config.blast_evalue),
            "-max_target_seqs", str(self.config.max_blast_hits),
            "-num_threads", str(self.config.blast_threads),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return [BlastResult(
                query=gene_name, subject="ERROR",
                identity=0, alignment_length=0,
                evalue=0, bitscore=0,
                subject_title=f"BLAST failed: {e}",
            )]

        results = []
        if output_file.exists():
            for line in output_file.read_text().strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 6:
                    results.append(BlastResult(
                        query=parts[0],
                        subject=parts[1],
                        identity=float(parts[2]),
                        alignment_length=int(parts[3]),
                        evalue=float(parts[4]),
                        bitscore=float(parts[5]),
                        subject_title=parts[6] if len(parts) > 6 else "",
                    ))

        return results

    def build_phylogeny(
        self,
        gene_name: str,
        sequences: dict[str, str],
    ) -> PhylogenyResult:
        """Build a phylogenetic tree for a gene across multiple sequences.

        Args:
            gene_name: name of the gene
            sequences: {sequence_name: sequence_string}
        """
        work = self._work_dir()
        input_file = work / f"{gene_name}_seqs.fasta"
        aligned_file = work / f"{gene_name}_aligned.fasta"
        tree_file = work / f"{gene_name}_tree.nwk"

        # Write sequences
        with open(input_file, "w") as f:
            for name, seq in sequences.items():
                f.write(f">{name}\n{seq}\n")

        # MAFFT alignment
        try:
            subprocess.run(
                ["mafft", "--auto", "--thread",
                 str(self.config.mafft_threads), str(input_file)],
                check=True, capture_output=True, timeout=300,
                stdout=open(aligned_file, "w"),
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return PhylogenyResult(
                gene_name=gene_name,
                n_sequences=len(sequences),
                newick=f"(error: MAFFT failed: {e})",
            )

        # FastTree
        try:
            result = subprocess.run(
                ["FastTree", "-nt", "-gtr", str(aligned_file)],
                check=True, capture_output=True, timeout=300,
            )
            newick = result.stdout.decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return PhylogenyResult(
                gene_name=gene_name,
                n_sequences=len(sequences),
                newick=f"(error: FastTree failed: {e})",
            )

        return PhylogenyResult(
            gene_name=gene_name,
            n_sequences=len(sequences),
            newick=newick,
        )

    def compute_ani(
        self,
        contigs_a: dict[str, str],
        contigs_b: dict[str, str],
        name_a: str = "set_a",
        name_b: str = "set_b",
    ) -> ANIResult:
        """Compute ANI between two sets of contigs using minimap2.

        Args:
            contigs_a: {contig_name: sequence} for set A
            contigs_b: {contig_name: sequence} for set B
        """
        work = self._work_dir()
        fasta_a = work / f"{name_a}.fasta"
        fasta_b = work / f"{name_b}.fasta"

        with open(fasta_a, "w") as f:
            for name, seq in contigs_a.items():
                f.write(f">{name}\n{seq}\n")
        with open(fasta_b, "w") as f:
            for name, seq in contigs_b.items():
                f.write(f">{name}\n{seq}\n")

        # minimap2 all-vs-all
        try:
            result = subprocess.run(
                ["minimap2", "-t", str(self.config.minimap2_threads),
                 "-cx", "asm20", str(fasta_a), str(fasta_b)],
                check=True, capture_output=True, timeout=300,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return ANIResult(
                set_a_name=name_a, set_b_name=name_b,
                ani=0.0, n_aligned=0, coverage=0.0,
            )

        # Parse PAF output for ANI estimation
        total_identity = 0.0
        total_aligned = 0
        for line in result.stdout.decode().strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 12:
                matches = int(parts[9])
                block_len = int(parts[10])
                if block_len > 0:
                    total_identity += matches
                    total_aligned += block_len

        size_a = sum(len(s) for s in contigs_a.values())
        size_b = sum(len(s) for s in contigs_b.values())
        shorter = min(size_a, size_b)

        ani = (total_identity / total_aligned * 100) if total_aligned > 0 else 0.0
        coverage = (total_aligned / shorter) if shorter > 0 else 0.0

        return ANIResult(
            set_a_name=name_a,
            set_b_name=name_b,
            ani=round(ani, 2),
            n_aligned=total_aligned,
            coverage=round(coverage, 4),
        )

    def detect_mobile_elements(
        self,
        contig_name: str,
        sequence: str,
        prokka_gff: Optional[str] = None,
    ) -> MobileElementResult:
        """Scan a contig for mobile element hallmarks.

        Uses gene annotations if available, falls back to sequence-based
        detection for phage hallmarks, plasmid backbone, IS elements.
        """
        hallmarks = []
        element_type = "none"
        confidence = 0.0
        evidence_parts = []

        # Check annotations if available
        if prokka_gff:
            gff_lower = prokka_gff.lower()

            # Phage hallmarks
            phage_markers = [
                "terminase", "capsid", "portal", "tail fiber",
                "tail protein", "baseplate", "head", "phage",
                "integrase", "recombinase",
            ]
            for marker in phage_markers:
                if marker in gff_lower:
                    hallmarks.append(marker)

            if len(hallmarks) >= 3:
                element_type = "prophage"
                confidence = min(0.5 + len(hallmarks) * 0.1, 0.95)
                evidence_parts.append(f"Phage hallmarks: {', '.join(hallmarks)}")

            # Plasmid hallmarks
            plasmid_markers = ["repa", "repb", "mob", "tra ", "traf", "relaxase"]
            plasmid_hits = [m for m in plasmid_markers if m in gff_lower]
            if plasmid_hits and element_type == "none":
                hallmarks.extend(plasmid_hits)
                element_type = "plasmid"
                confidence = min(0.4 + len(plasmid_hits) * 0.15, 0.9)
                evidence_parts.append(f"Plasmid markers: {', '.join(plasmid_hits)}")

            # IS elements / transposons
            is_markers = ["transposase", "insertion sequence", "is element", "tnpa"]
            is_hits = [m for m in is_markers if m in gff_lower]
            if is_hits and element_type == "none":
                hallmarks.extend(is_hits)
                element_type = "transposon"
                confidence = min(0.3 + len(is_hits) * 0.2, 0.85)
                evidence_parts.append(f"IS/transposon markers: {', '.join(is_hits)}")

        # Sequence-based: check for integration sites (tRNA at boundaries)
        # Simple heuristic: look for tRNA-like sequences near contig ends
        if len(sequence) > 1000:
            end_5 = sequence[:500].upper()
            end_3 = sequence[-500:].upper()
            # tRNA consensus: look for CCA at 3' end
            if "CCA" in end_5[-10:] or "CCA" in end_3[-10:]:
                evidence_parts.append("Possible tRNA integration site at boundary")
                if element_type == "prophage":
                    confidence = min(confidence + 0.1, 0.95)

        return MobileElementResult(
            contig_name=contig_name,
            element_type=element_type,
            confidence=round(confidence, 2),
            hallmark_genes=hallmarks,
            evidence="; ".join(evidence_parts) if evidence_parts else "No mobile element features detected",
        )

    def map_reads_to_region(
        self,
        contigs: dict[str, str],
        reads_path: Path,
    ) -> dict:
        """Map reads to specific contigs and report coverage.

        Returns: {contig_name: {"n_reads": int, "mean_depth": float}}
        """
        work = self._work_dir()
        ref_file = work / "region_ref.fasta"
        bam_file = work / "region_mapped.bam"

        with open(ref_file, "w") as f:
            for name, seq in contigs.items():
                f.write(f">{name}\n{seq}\n")

        try:
            # Map with minimap2
            map_cmd = (
                f"minimap2 -t {self.config.minimap2_threads} -ax map-ont "
                f"{ref_file} {reads_path} | "
                f"samtools view -F 0x904 -b | "
                f"samtools sort -o {bam_file}"
            )
            subprocess.run(map_cmd, shell=True, check=True, capture_output=True, timeout=600)
            subprocess.run(["samtools", "index", str(bam_file)], check=True, capture_output=True)

            # Get coverage stats
            result = subprocess.run(
                ["samtools", "idxstats", str(bam_file)],
                check=True, capture_output=True,
            )

            stats = {}
            for line in result.stdout.decode().strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 3 and parts[0] != "*":
                    stats[parts[0]] = {
                        "n_reads": int(parts[2]),
                        "length": int(parts[1]),
                    }

            return stats

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return {"error": str(e)}

    def mini_reassemble(
        self,
        reads_path: Path,
        region_contigs: dict[str, str],
    ) -> dict:
        """Targeted re-assembly of reads mapping to a genomic region.

        1. Extract reads that map to the region
        2. Run Flye on just those reads
        3. Return the assembly result
        """
        work = self._work_dir()
        ref_file = work / "region_ref.fasta"
        mapped_reads = work / "region_reads.fastq"
        assembly_dir = work / "reassembly"

        with open(ref_file, "w") as f:
            for name, seq in region_contigs.items():
                f.write(f">{name}\n{seq}\n")

        try:
            # Extract reads mapping to region
            extract_cmd = (
                f"minimap2 -t {self.config.minimap2_threads} -ax map-ont "
                f"{ref_file} {reads_path} | "
                f"samtools view -F 0x904 -b | "
                f"samtools fastq - > {mapped_reads}"
            )
            subprocess.run(extract_cmd, shell=True, check=True, capture_output=True, timeout=600)

            # Check if we got enough reads
            n_reads = sum(1 for _ in open(mapped_reads)) // 4
            if n_reads < 10:
                return {"error": f"Only {n_reads} reads mapped — too few for re-assembly"}

            # Run Flye
            flye_cmd = [
                "flye", "--nano-raw", str(mapped_reads),
                "--out-dir", str(assembly_dir),
                "--genome-size", self.config.flye_genome_size,
                "--threads", str(self.config.flye_threads),
                "--min-overlap", "1000",
            ]
            subprocess.run(flye_cmd, check=True, capture_output=True, timeout=1200)

            # Parse results
            assembly_fasta = assembly_dir / "assembly.fasta"
            if assembly_fasta.exists():
                contigs = {}
                current_name = None
                current_seq = []
                for line in open(assembly_fasta):
                    if line.startswith(">"):
                        if current_name:
                            contigs[current_name] = len("".join(current_seq))
                        current_name = line.strip()[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())
                if current_name:
                    contigs[current_name] = len("".join(current_seq))

                return {
                    "n_reads": n_reads,
                    "n_contigs": len(contigs),
                    "contigs": contigs,
                    "total_size": sum(contigs.values()),
                }
            else:
                return {"error": "Flye produced no output"}

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Elder API tool definitions (for Claude tool-use)
# ---------------------------------------------------------------------------

ELDER_TOOLS = [
    {
        "name": "blast_gene",
        "description": "BLAST a gene sequence against NCBI nr/nt to identify its closest relatives. Returns top hits with identity, taxonomy, and descriptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Contig containing the gene"},
                "gene_name": {"type": "string", "description": "Name of the gene to BLAST"},
            },
            "required": ["contig_name", "gene_name"],
        },
    },
    {
        "name": "build_phylogeny",
        "description": "Build a phylogenetic tree for a gene across multiple contigs plus reference sequences. Uses MAFFT alignment + FastTree.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_name": {"type": "string", "description": "Name of the gene"},
                "contig_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Contigs containing variants of this gene",
                },
            },
            "required": ["gene_name", "contig_names"],
        },
    },
    {
        "name": "compute_ani",
        "description": "Compute Average Nucleotide Identity between two sets of contigs. >95% ANI = same species, >99% ANI = same strain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contigs_a": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "First set of contig names",
                },
                "contigs_b": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Second set of contig names",
                },
            },
            "required": ["contigs_a", "contigs_b"],
        },
    },
    {
        "name": "detect_mobile_elements",
        "description": "Scan a contig for mobile genetic element hallmarks: prophage (terminase, capsid, integrase), plasmid (repA, mob, tra), transposon (transposase, IS elements), or genomic island (integrase + tRNA boundary).",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_name": {"type": "string", "description": "Contig to scan"},
            },
            "required": ["contig_name"],
        },
    },
    {
        "name": "map_reads_to_region",
        "description": "Map original reads to specific contigs and report coverage. Useful for checking if two 'redundant' contigs have shared read support.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Contigs to map reads to",
                },
            },
            "required": ["contig_names"],
        },
    },
    {
        "name": "mini_reassemble",
        "description": "Extract reads mapping to a genomic region and run targeted Flye re-assembly. Use when SCG variants are ambiguous — if re-assembly produces one contig, variants were assembly artifacts; if two, they're real population diversity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contig_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Contigs defining the region to re-assemble",
                },
            },
            "required": ["contig_names"],
        },
    },
]


# ---------------------------------------------------------------------------
# Elder investigation workflow
# ---------------------------------------------------------------------------

def identify_redundant_markers(
    community: CommunityProfile,
    identities: dict[str, ContigIdentity],
) -> list[tuple[str, list[str]]]:
    """Find redundant single-copy markers within a community.

    Returns: [(marker_name, [contig_names_carrying_it])]
    """
    marker_to_contigs: dict[str, list[str]] = {}
    for member_name in community.members:
        c = identities.get(member_name)
        if c and c.marker_genes:
            for gene in c.marker_genes:
                marker_to_contigs.setdefault(gene, []).append(member_name)

    # Only return markers present on 2+ contigs
    return [
        (marker, contigs)
        for marker, contigs in marker_to_contigs.items()
        if len(contigs) > 1
    ]
