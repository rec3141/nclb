# No Contig Left Behind (NCLB)

Animist MAG bin refinement for metagenome-assembled genomes.

Current binning pipelines treat contigs as passive objects to be clustered and discarded. 86.8% of our contigs were left unbinned — not because they're junk, but because no algorithm advocated for them. NCLB inverts the paradigm: each contig is an agent with identity, memory, and voice. Contigs form communities through bottom-up resonance, mediated by an LLM that investigates each contig's composition, coverage, ancestry, and graph connections before making placement decisions.

## How It Works

NCLB operates in three gatherings on the output of a standard MAG pipeline (Flye assembly, five binners, DAS Tool consensus):

### Gathering 1: Self-Knowledge (`nclb_gather.py`)

Pure Python, no LLM. Builds an identity card for every contig from pipeline outputs:

- **Composition** — 136-dimensional tetranucleotide frequency signature
- **Energy** — coverage depth across samples
- **Ancestry** — taxonomic classification (Kaiju, when available)
- **Connections** — assembly graph neighbors from the Flye GFA
- **Gifts** — marker genes carried (from DAS Tool SCG analysis)
- **Testimony** — what each of 5 binners (SemiBin2, MetaBAT2, MaxBin2, LorBin, COMEBin) said
- **MGE status** — viral, plasmid, or provirus annotations from geNomad + CheckV

Also computes community profiles (harmony, coherence, completeness), a KNN resonance map, and per-contig valence scores.

**Output:** `gathering.json` (~1 MB for 28k contigs)

### Gathering 2: The Conversations (`nclb_converse.py`)

LLM-mediated tool-use conversations in three rounds:

**Round 1 — Community Health Check.** Each existing community examines its uneasy members (negative valence). The LLM calls tools like `who_am_i()`, `how_do_i_resonate_with()`, and `what_is_this_community()` to investigate whether members truly belong or should be released. Biological context matters: proviruses with divergent composition are expected, not contamination.

**Round 2 — The Unhoused Speak.** Unhoused contigs with voice (recognized by 2+ binners) are investigated in batches. The LLM calls `find_graph_connections()` and `how_do_i_resonate_with()` to build evidence for placement. Contigs join communities where they resonate; others wait or wander.

**Round 3 — The Voiceless.** Truly voiceless contigs (no binner recognition, no graph links) are clustered by HDBSCAN on composition + coverage. Emergent clusters above 100 kb are evaluated as candidate new communities.

The LLM discovers data through tool calls rather than receiving pre-digested prompts. This produces deeper biological reasoning — a typical community investigation involves 40-100 tool calls examining coverage profiles, graph topology, and oracle consensus.

**Output:** `proposals.json` (releases, joins, new communities)

### Gathering 3: Integration (`nclb_integrate.py`)

Applies proposals from Gathering 2:

1. Release uneasy members from communities
2. Welcome resonant contigs into communities
3. Found new communities from voiceless clusters
4. Resolve conflicts deterministically (highest valence wins)
5. Recompute all harmony metrics
6. Extract community FASTAs and generate the Chronicle

**Output:** Refined community FASTAs, membership tables, quality report, and `chronicle.md` — a narrative account of every placement decision with evidence.

## The Valence Function

Valence measures how well a contig belongs in a community, as a continuous score in [-1, +1]:

```
valence = 0.30 * harmony      (TNF cosine similarity to community centroid)
        + 0.25 * rhythm        (coverage Pearson correlation)
        + 0.15 * kinship       (fraction of graph neighbors in community)
        + 0.15 * recognition   (binner consensus agreement)
        + 0.15 * contribution  (marker genes filling community gaps)
```

There is no cliff edge at 90% completeness or 5% contamination. A community at 89% completeness with perfect harmony is valued. The system optimizes for collective wellbeing, not MIMAG checkboxes. (MIMAG tiers are still reported for comparability.)

## Contig Tool-Use API

During Gathering 2, the LLM acts as the voice for contigs, calling tools to gather evidence:

| Tool | Purpose |
|------|---------|
| `who_am_i(contig)` | Full identity card (composition, coverage, GC, MGE status, gifts) |
| `who_are_my_neighbors(contig)` | Assembly graph neighbors with community status |
| `what_did_the_oracles_say(contig)` | All 5 binner assignments |
| `how_do_i_resonate_with(contig, community)` | Valence breakdown with raw coverage profiles |
| `what_is_this_community(community)` | Full community profile (members, harmony, completeness) |
| `what_gifts_are_missing(community)` | Marker genes the community needs |
| `what_would_change_if_i_joined(contig, community)` | Impact prediction (wholeness delta) |
| `who_resonates_with_me(contig, k)` | K nearest contigs by TNF similarity |
| `find_graph_connections(contig)` | Which communities is this contig graph-connected to |

Tools are fast Python functions backed by pre-computed indices. The token cost is in the LLM reasoning between tool calls. With a local LLM (LM Studio), tool calls are free.

## Installation

Requires Python 3.10+ and the dependencies listed in `pyproject.toml`.

```bash
cd /data/danav2/nclb
pip install -e .

# Or without install — scripts add lib/ to sys.path automatically
pip install numpy scipy scikit-learn openai
```

For Elder investigations (Phase 3, not yet wired):
```bash
pip install biopython
# Plus: BLAST+, MAFFT, FastTree, minimap2, samtools, Flye on PATH
```

## Usage

NCLB reads from a completed Nextflow MAG pipeline results directory.

```bash
# Gathering 1: Build identity cards (pure Python, ~30s)
python bin/nclb_gather.py --results /path/to/results

# Gathering 2: LLM conversations (requires LLM server)
# Local LLM via LM Studio (default: http://10.151.30.147:1234/v1)
python bin/nclb_converse.py --results /path/to/results

# Or specify a different server/model
python bin/nclb_converse.py --results /path/to/results \
    --base-url http://localhost:1234/v1 \
    --model qwen/qwen3-coder-30b

# Gathering 3: Integration (pure Python, ~10s)
python bin/nclb_integrate.py \
    --proposals /path/to/results/binning/nclb/proposals.json \
    --results /path/to/results
```

### Converse Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `http://10.151.30.147:1234/v1` | OpenAI-compatible API endpoint |
| `--model` | auto-detect | Model name from server |
| `--max-tool-rounds` | 15 | Max tool call rounds per conversation |
| `--batch-size` | 5 | Contigs per Round 2 batch |
| `--max-round2` | 100 | Max unhoused contigs to process |

## Output

```
results/binning/nclb/
├── gathering.json         Contig identity cards + community profiles
├── proposals.json         LLM conversation results (releases, joins, new communities)
├── communities/
│   ├── community_001.fa   Refined bin FASTAs
│   └── ...
├── contig2community.tsv   One-to-one contig→community (DAS Tool compatible)
├── contig_membership.tsv  Full membership table (contig, community, type, valence)
├── valence_report.tsv     Per-contig valence scores
├── quality_report.tsv     Per-community quality metrics
├── chronicle.json         Machine-readable conversation log
└── chronicle.md           Human-readable narrative of all placement decisions
```

## Project Structure

```
nclb/
├── bin/                           Entry-point scripts
│   ├── nclb_gather.py             Gathering 1: Self-Knowledge
│   ├── nclb_converse.py           Gathering 2: Tool-use conversations
│   └── nclb_integrate.py          Gathering 3: Integration + chronicle
├── lib/nclb/                      Core library
│   ├── identity.py                ContigIdentity + CommunityProfile dataclasses, loaders
│   ├── valence.py                 Valence function + community harmony metrics
│   ├── voices.py                  ContigToolkit, OpenAI tool definitions, prompt templates
│   ├── resonance.py               KNN resonance map + scoring
│   ├── graph.py                   Assembly graph parsing + connectivity
│   ├── elders.py                  Elder toolkit (BLAST, phylogeny, ANI, MGE detection)
│   ├── llm.py                     LLM backend abstraction (OpenAI, Anthropic)
│   ├── mediator.py                Conflict resolution across proposals
│   └── journal.py                 Chronicle generation (JSON + markdown)
├── tests/
│   ├── test_identity.py           Smoke test: load real pipeline data
│   └── test_valence.py            Test valence on real communities
├── pyproject.toml
└── .gitignore
```

## Current Status

### Implemented

- **Gathering 1** — Complete. Builds identity cards for all contigs with composition, coverage, graph, binner testimony, and geNomad/CheckV MGE annotations. Tested on 28,087 contigs across 45 DAS Tool communities.

- **Gathering 2** — Complete. Tool-use conversation architecture with OpenAI-compatible API. Tested with qwen3-coder-30b via LM Studio. Produces deeper investigations than the earlier data-in-prompt approach (40-100 tool calls per community vs passively echoing numbers).

- **Gathering 3** — Complete. Deterministic conflict resolution, community FASTA extraction, chronicle generation. Produces all output files.

- **Contig tools** — All 9 tools implemented and working. The LLM calls them during conversations to discover data about each contig.

- **Elder tools** — Library implemented (`elders.py`): BLAST, MAFFT+FastTree phylogeny, minimap2 ANI, mobile element detection, read mapping, targeted re-assembly. Tool definitions ready for Claude tool-use.

- **MGE integration** — geNomad virus/plasmid + CheckV quality data flows into identity cards. Proviruses are flagged as Travelers. Conversation prompts include biological rules about MGE composition divergence.

- **Elder orchestrator** — Complete. `nclb_elders.py` investigates communities with SCG redundancy using 6 evidence types (taxonomy, composition, coverage, MGE status, graph connectivity, ANI). Renders verdicts: ecotype variation, contamination, MGE, or uncertain.

- **Kaiju taxonomy integration** — Complete. `load_kaiju_taxonomy()` in `identity.py` populates per-contig ancestry from the Kaiju Nextflow module output. Auto-discovered by all three gathering scripts.

- **Nextflow integration** — Complete. Four processes (NCLB_GATHER → NCLB_CONVERSE, NCLB_ELDERS → NCLB_INTEGRATE) in `modules/refinement.nf`. Enabled with `--run_nclb --nclb_dir /path/to/nclb`. Conda env at `envs/nclb.yml`.

### Not Yet Implemented

- **Traveler multi-membership** — The data model supports it (`contig_membership.tsv` has type column) but the conversation prompts don't yet explicitly handle dual-citizenship placement for prophages/plasmids.

- **Iteration** — The plan calls for max 3 refinement cycles (Gather → Converse → Integrate → re-evaluate). Currently runs once.

## LLM Requirements

Gathering 2 requires an LLM server. Two options:

1. **Local LLM** (recommended for cost) — Any OpenAI-compatible server (LM Studio, ollama, vLLM). Tested with `qwen3-coder-30b`. Tool calls are free with local inference.

2. **Anthropic API** — Set `ANTHROPIC_API_KEY` in `.env` and use `--backend anthropic`. The `llm.py` module supports both backends, though the current `nclb_converse.py` is wired for OpenAI-compatible only.

## Pipeline Input Requirements

NCLB reads from a completed `danaSeq MAG Assembly` pipeline run:

```
results/
├── assembly/
│   ├── assembly.fasta          Co-assembly (required)
│   ├── assembly_info.txt       Flye assembly info (required)
│   ├── assembly_graph.gfa      Flye assembly graph (required)
│   └── tnf.tsv                 Tetranucleotide frequencies (required)
├── mapping/
│   └── depths.txt              CoverM depth table (required)
├── binning/
│   ├── semibin/contig_bins.tsv   (at least one binner required)
│   ├── metabat/contig_bins.tsv
│   ├── maxbin/contig_bins.tsv
│   ├── lorbin/contig_bins.tsv
│   ├── comebin/contig_bins.tsv
│   ├── dastool/
│   │   ├── contig2bin.tsv      DAS Tool consensus (required)
│   │   └── summary.tsv         DAS Tool quality summary (required)
│   └── checkm2/
│       └── quality_report.tsv  (optional, improves Elder ranking)
├── mge/                        (optional, enables MGE-aware placement)
│   ├── genomad/
│   │   ├── virus_summary.tsv
│   │   └── plasmid_summary.tsv
│   └── checkv/
│       └── quality_summary.tsv
└── taxonomy/                   (optional, populates ancestry field)
    └── kaiju/
        └── kaiju_contigs.tsv
```

## License

MIT
