# Later physical case: mixed BDC/amino-BDC in a MOF-5-family framework

This directory contains a **recipe, not executable atomic input data**. Its purpose is to define a manageable next step after the synthetic integration fixture passes.

Use a MOF-5/IRMOF-1 parent framework with Zn4O nodes, BDC and 2-amino-BDC linkers, and a chosen 1:1 linker ratio. A primary report describes this mixed-linker material family: [Postsynthetic modification of mixed-linker metal–organic frameworks for ethylene oligomerization](https://pubs.rsc.org/en/content/articlehtml/2014/ra/c4ra10605a), DOI 10.1039/C4RA10605A. It supports the material-family choice, not the synthetic energies or a claim that the selected 50:50 composition was experimentally incorporated. Exclude the paper's subsequent Ni-functionalization step from this proposed initial case.

Before running it:

1. Obtain a verified parent CIF with a recorded source and checksum. Resolve solvent/guest removal, missing atoms, disorder, bonding and hydrogen conventions explicitly.
2. Obtain or generate validated BDC/amino-BDC and Zn4O construction blocks with attachment-site mappings. The SMILES in `recipe.json` specify linker chemistry; they do not supply validated three-dimensional building blocks.
3. Confirm the parent net and extract its actual substitution sites. Use a cell or supercell with an even number of Zn4O units so an exact 1:1 mixture is integral. The synthetic 12-site graphs do not define this physical cell.
4. Fix the initial amino-linker orientation convention for reproducibility. Then, if orientation affects the scientific question, expand the configuration variables to include it; binary linker identity alone does not describe every orientation.
5. Build a small diverse arrangement set at the same composition, relax using a declared protocol, and record comparable periodic energies. Use an ML potential only after checking its applicability to this chemistry; label failed constructions and calculations explicitly.
6. Replace the synthetic training labels with those calculated energies in a separate project. Refit and validate with tolerances chosen for the physical task. Record units, normalization, spin/charge settings, computational parameters and geometry identity.

[PORMAKE's official repository](https://github.com/Sangwon91/PORMAKE) documents topology/building-block construction and assignment of different blocks to individual slots. Verify the API and database contents in the installed version instead of assuming a particular block ID.

This case initially exercises a single topology, so it tests physical arrangement handling rather than topology competition. Nominal Zn(II) nodes are a convenient closed-shell starting point; this is not designed to demonstrate a strong-correlation correction or quantum advantage. Add a separately specified open-shell benchmark after the construction and energy-reference conventions work reliably.
