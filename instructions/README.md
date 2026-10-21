MOF pipeline implementation handoff

Prepared 21 October 2026. This package contains implementation specifications, not a runnable scientific pipeline. All proposed tool names are project interfaces to implement; they are not claims about existing third-party APIs.

Goal: rank mixed-linker or mixed-metal arrangements and candidate framework topologies at fixed composition using static-lattice energies, calibrated configurational models, and optional validated embedded electronic corrections. Finite-temperature ensembles and quantum MCMC are outside scope.

Give every coding agent 00_SHARED_CONTRACT.md, one numbered task brief, and the relevant records/interfaces from adjacent modules. A separate integrator owns shared schemas and orchestration. Tasks 01–09 describe scientific modules; 10_AGENT_ARCHITECTURE.md describes runtime agents and implementation order. A coding agent that implements a module is different from an LLM role invoked during a scientific run.

Recommended first case: one chemical family, two supplied topologies, binary substitutions, a fixed feasible composition, a small cell allowing exhaustive reference enumeration, and a second cell for transfer tests. Choose actual structures and numerical tolerances before launching scientific calculations. Do not invent structures, energies, access credentials or tolerances.

Deliverables from each coding agent: working module, typed interface, dependency/environment record, scientific acceptance checks, one reproducible example, and a short handoff stating implemented, unsupported and blocked capabilities. Stub calculators may test plumbing, but outputs must be labelled SYNTHETIC and excluded from scientific ranking.

The package includes project_spec.template.json, a deliberately incomplete input template, and SOURCES.md with primary documentation. Null configuration entries require scientific or infrastructure choices; they are not runnable defaults.

