Task 09 — Structural validation, competition and final report

Assignment
Implement src/validation, src/ranking and report generation. Apply lightweight validation throughout the workflow and expensive checks to finalists.

Inputs
All candidate artifacts, final compatible energy group, uncertainty records, available reference phases and experimental data, benchmark suite and search-coverage ledger.

Implement
1. Use MOFChecker for declared sanity checks, CrystalNets/MOFid for topology/identity, and Zeo++ for geometric pore descriptors. Wrap each tool with version, parameters, parseable status and raw evidence. Tool success is not proof of synthesizability.
2. Apply post-relaxation connectivity and residual-force/stress checks. Add appropriate phonon/elastic or targeted stability calculations for finalist claims. Static-lattice ranking is the main scope; do not add finite-temperature ensembles.
3. Rank only energy-compatible candidates. Report unresolved differences within numerical/model uncertainty as ties. If high-level corrections are unavailable for a plausible competitor, mark the corrected ranking incomplete.
4. Check relevant phase-separated reference combinations at the same overall composition and compatible energy level. Solve the reference mixture balance explicitly. Report incomplete reference coverage rather than claiming absolute stability.
5. Compare predicted PXRD and, where a validated calculation protocol exists, NMR observables to actual referenced measurements. Use a comparison protocol with instrumental effects/tolerances. Paramagnetic NMR needs appropriate physics; do not promise it from a generic structure parser.
6. Benchmark modules against Kang arrangements, Cai topology competition, and the published Fe–MOF-74 electronic workflow, with exact paper/data versions and settings. Distinguish reproduction of a paper's calculation from agreement with experiment; retain known counterexamples.
7. Demonstrate one integrated chemical family. Include ablations: fixed topology versus joint search; heuristic versus fitted costs; no versus validated embedding correction; classical versus quantum optimizer; deterministic versus agent-assisted orchestration. Choose ablations that test actual claims.
8. Export best-supported structures, close competitors, normalized energy table, uncertainty/sensitivity, full assumptions, exhausted search/budget, validation outcomes and reproducibility instructions.

Proposed tools
validate_structure(geometry_id, validation_protocol_id) -> ValidationReport
rank_candidates(candidate_ids, comparison_policy_id) -> RankingReport
evaluate_reference_mixtures(ranking_id, reference_set_id) -> CompetitionReport
generate_report(project_snapshot_id) -> ReportArtifacts

Acceptance
Wrong units/composition/fidelity prevent pooling. A missing reference produces an incomplete result, not a stability certificate. Simulated observations are labelled simulated. Exact-solver benchmarks, chemical validation and experiment comparisons remain separate. Every plotted/reported number traces to a stored artifact.

Agent/API use
An independent reviewer reads artifacts and flags unsupported claims; a report agent may explain verified tables. Numerical tables are produced by code and cannot be rewritten by the LLM.

