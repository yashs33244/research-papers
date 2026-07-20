# Research Papers

LaTeX sources, compiled PDFs, and (where applicable) full experimental harnesses for my research papers on LLM agents, context engineering, reliability, and model architecture.

Rendered index with PDFs: https://yashs33244.github.io/research-papers/

| Paper | Year | Directory |
|---|---|---|
| Size or Architecture? A Parameter-Matched Comparison of a Transformer and the Dragon Hatchling (full harness + code) | 2026 | [`nano-gpt-vs-bdh/`](./nano-gpt-vs-bdh) |
| The Compaction Half-Life: Measuring Fact Survival Under Iterated Context Summarization | 2026 | [`compaction-half-life/`](./compaction-half-life) |
| What Ships: A Practitioner's Catalogue of Applied AI Product Patterns in Startups, 2024-2026 | 2026 | [`applied-ai-patterns/`](./applied-ai-patterns) |
| Defense in Depth for Language-Model Applications | 2026 | [`llm-defense-in-depth/`](./llm-defense-in-depth) |
| From Agent Loops to Structured Graphs (code: [loops_vs_agents](https://github.com/yashs33244/loops_vs_agents)) | 2026 | [`graph-scheduled-agents/`](./graph-scheduled-agents) |

## Reproducibility

`compaction-half-life/experiment/` contains the complete pre-registered harness: chain generation, model calls, scoring, analysis, and every raw model response (105 chain calls + probes, results.csv, bootstrap analysis). Every number in the paper is computed from these files.

`nano-gpt-vs-bdh/experiment/` contains from-scratch implementations of both a GPT-2-style Transformer and BDH (the Dragon Hatchling), a shared training/eval harness, both trained checkpoints, all generated completions, and both orderings of every LLM-judge verdict for both the parameter-matched and the 2x-size runs. `gen_paper_assets.py` emits every number and table in the paper from these raw files.

## Building

Each paper compiles standalone with [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd <paper-dir> && tectonic paper.tex
```
