# Research Papers

LaTeX sources, compiled PDFs, and (where applicable) full experimental harnesses for my research papers on LLM agents, context engineering, and reliability.

Rendered index with PDFs: https://yashs33244.github.io/research-papers/

| Paper | Year | Directory |
|---|---|---|
| The Compaction Half-Life: Measuring Fact Survival Under Iterated Context Summarization | 2026 | [`compaction-half-life/`](./compaction-half-life) |
| What Ships: A Practitioner's Catalogue of Applied AI Product Patterns in Startups, 2024-2026 | 2026 | [`applied-ai-patterns/`](./applied-ai-patterns) |
| Defense in Depth for Language-Model Applications | 2026 | [`llm-defense-in-depth/`](./llm-defense-in-depth) |
| From Agent Loops to Structured Graphs (code: [loops_vs_agents](https://github.com/yashs33244/loops_vs_agents)) | 2026 | [`graph-scheduled-agents/`](./graph-scheduled-agents) |

## Reproducibility

`compaction-half-life/experiment/` contains the complete pre-registered harness: chain generation, model calls, scoring, analysis, and every raw model response (105 chain calls + probes, results.csv, bootstrap analysis). Every number in the paper is computed from these files.

## Building

Each paper compiles standalone with [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd <paper-dir> && tectonic paper.tex
```
