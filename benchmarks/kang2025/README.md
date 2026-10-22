# Kang et al. (2025) design benchmark

Inputs from Kang, Kim & Kim, *Quantum Computing Based Design of Multivariate Porous Materials*, ACS Cent. Sci. 2025
(doi:10.1021/acscentsci.5c00918; arXiv:2502.06339v2): site graphs and edge distances (Table S3, Fig. S2) and linker
characteristic lengths (Table S2). `public` is what A2 sees. `reference` is never sent to the model:
- the experimental arrangement (the paper's model ground states, which it reports as corresponding to the experimental structures);
- the paper's VQE ground-state probability;
- `kang_control`, the paper's own weighting choices expressed in the term catalogue, used to check the compiler.

SIOC-COF2 is not included. Its kgm graph could not be reconstructed to reproduce the paper's spectrum.

    PYTHONPATH=src python3 -m mofpipe agents-design-bench --controls controls.yaml --models qwen/qwen3.8-flash --repeats 3
