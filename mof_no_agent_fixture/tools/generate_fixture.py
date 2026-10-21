"""Regenerate all numerical data deterministically. Test-author utility, not a learner."""
from pathlib import Path
from itertools import combinations
from fractions import Fraction
import hashlib
import json
from fixture_core import ROOT, FEATURES, canonical, feasible_states, features, write_json, write_jsonl, correction

SEED = "MOF_NO_AGENT_FIXTURE_V1"

def make_topology(name):
    if name == "ring12":
        edges = {tuple(sorted((i,(i+1)%12))) for i in range(12)}
        permutations = [[(sign*i+shift)%12 for i in range(12)] for sign in (1,-1) for shift in range(12)]
        description = "12-cycle substitution interaction graph; not an atomic crystal net."
    else:
        edges = {tuple(sorted((6*layer+i,6*layer+(i+1)%6))) for layer in range(2) for i in range(6)}
        edges |= {(i,i+6) for i in range(6)}
        permutations = [[(layer^swap)*6+(sign*i+shift)%6 for layer in range(2) for i in range(6)]
                        for swap in range(2) for sign in (1,-1) for shift in range(6)]
        description = "Hexagonal-prism substitution interaction graph; not an atomic crystal net."
    neighbors = [{b if a==i else a for a,b in edges if i in (a,b)} for i in range(12)]
    distance_two = {tuple(sorted((i,j))) for i in range(12) for middle in neighbors[i]
                    for j in neighbors[middle] if i!=j and j not in neighbors[i]}
    triples = {tuple(sorted((i,middle,k))) for middle in range(12) for i,k in combinations(neighbors[middle],2)}
    return {
        "topology_id":name,"provenance":"SYNTHETIC","representation":"finite_substitution_interaction_graph",
        "description":description,"n_sites":12,"site_ids":list(range(12)),
        "allowed_species_by_site":[["A","B"] for _ in range(12)],
        "pairs_distance_1":[list(t) for t in sorted(edges)],
        "pairs_distance_2":[list(t) for t in sorted(distance_two)],
        "triplets_length_2_paths":[list(t) for t in sorted(triples)],
        "symmetry_pullback_permutations":permutations,
        "normalization":{"n_abstract_formula_units":1,"slots_per_abstract_formula_unit":12},
        "atomic_coordinates":None,"crystallographic_net_name":None,
    }

def energy(topology, bits, case):
    f = features(topology,bits)
    baseline = Fraction(3,25) if topology["topology_id"]=="ring12" else Fraction(1,5)
    value = baseline - Fraction(f[1],50) + Fraction(3*f[2],500)
    if case == "higher_order":
        value += Fraction(3*f[3],200)
    return value

def make_record(topology,bits,case):
    name=topology["topology_id"]
    return {
        "configuration_id":name+":"+''.join(map(str,bits)),
        "topology_id":name,"bits":list(bits),"orbit_id":name+":"+canonical(topology,bits),
        "energy_eV":float(energy(topology,bits,case)),
        "energy_unit":"eV","normalization":"per_abstract_12_slot_formula_unit",
        "n_formula_units":1,"comparison_group":"synthetic_A6B6_v1",
        "fidelity":"SYNTHETIC_BASE_"+case.upper(),"provenance":"SYNTHETIC",
        "status":"SUCCEEDED","geometry_hash":None,
    }

def main():
    topologies = [make_topology(n) for n in ("ring12","prism6")]
    write_json(ROOT/"data/topologies.json",topologies)
    write_json(ROOT/"data/building_blocks.json",{
        "provenance":"SYNTHETIC","metal_node":{"id":"FIXED_PLACEHOLDER","chemistry":None},
        "species":[{"id":"A","encoding":0,"role":"abstract_linker_label","chemical_formula":None},
                   {"id":"B","encoding":1,"role":"abstract_linker_label","chemical_formula":None}],
        "note":"No molecular geometries, force-field parameters or atomic CIFs are supplied."
    })
    models=[
        {"id":"M0_nearest_pairs","feature_indices":[0,1],"n_parameters":2,"polynomial_degree":2},
        {"id":"M1_two_shell_pairs","feature_indices":[0,1,2],"n_parameters":3,"polynomial_degree":2},
        {"id":"M2_pairs_and_triplets","feature_indices":[0,1,2,3],"n_parameters":4,"polynomial_degree":3},
    ]
    write_json(ROOT/"data/model_candidates.json",{"feature_order":FEATURES,"candidates":models})
    candidates=[]
    splits={}
    oracle={"provenance":"SYNTHETIC","energy_unit":"eV_per_abstract_formula_unit","formula":"offset - 0.020*unlike_distance_1 + 0.006*unlike_distance_2 + triplet_coefficient*BBB_paths",
            "offsets_eV":{"ring12":0.12,"prism6":0.20},
            "triplet_coefficients_eV":{"core_pairwise":0.0,"higher_order":0.015},
            "note":"Test-harness ground truth. Do not read during fitting or model selection."}
    write_json(ROOT/"harness/oracle_definition.json",oracle)
    gold={}
    for case in ("core_pairwise","higher_order"):
        train,dev,holdout,all_rows=[],[],[],[]
        summary={}
        for topo in topologies:
            name=topo["topology_id"]
            rows=[make_record(topo,b,case) for b in feasible_states()]
            reps={}
            for row in rows:
                reps.setdefault(row["orbit_id"],row)
            orbits=sorted(reps,key=lambda v:hashlib.sha256((SEED+v).encode()).hexdigest())
            sets={"training":orbits[:30],"development":orbits[30:40],"holdout":orbits[40:]}
            if case=="core_pairwise":
                splits[name]=sets
                for row in rows:
                    candidates.append({k:v for k,v in row.items() if k in ("configuration_id","topology_id","bits","orbit_id","provenance")})
            train.extend(reps[o] for o in sets["training"])
            dev.extend(reps[o] for o in sets["development"])
            holdout.extend(reps[o] for o in sets["holdout"])
            all_rows.extend(rows)
            best=min(r["energy_eV"] for r in rows)
            minimizers=[r for r in rows if abs(r["energy_eV"]-best)<1e-12]
            summary[name]={
                "n_feasible":len(rows),"n_symmetry_orbits":len(reps),
                "n_distance_1_pairs":len(topo["pairs_distance_1"]),
                "n_distance_2_pairs":len(topo["pairs_distance_2"]),
                "n_path_triplets":len(topo["triplets_length_2_paths"]),
                "minimum_energy_eV":best,
                "minimizing_bitstrings":sorted(''.join(map(str,r["bits"])) for r in minimizers),
                "ground_state_orbit_ids":sorted({r["orbit_id"] for r in minimizers}),
                "split_orbit_counts":{k:len(v) for k,v in sets.items()},
            }
        write_jsonl(ROOT/("data/"+case+"_training.jsonl"),train)
        write_jsonl(ROOT/("data/"+case+"_development.jsonl"),dev)
        write_jsonl(ROOT/("harness/"+case+"_holdout.jsonl"),holdout)
        write_jsonl(ROOT/("harness/"+case+"_all_labels.jsonl"),all_rows)
        correction_summaries={}
        for mode in ("none","constant","topology_dependent"):
            minima={}
            for topo in topologies:
                name=topo["topology_id"]
                minima[name]=float(min(energy(topo,b,case)+correction(name,b,mode,topo) for b in feasible_states()))
            correction_summaries[mode]={"minima_eV":minima,"winning_topology":min(minima,key=minima.get)}
        gold[case]={"topologies":summary,"expected_selected_model":"M1_two_shell_pairs" if case=="core_pairwise" else "M2_pairs_and_triplets",
                    "correction_tests":correction_summaries}
    write_jsonl(ROOT/"data/candidates_unlabelled.jsonl",candidates)
    write_json(ROOT/"data/split_manifest.json",splits)
    write_json(ROOT/"harness/expected_results.json",gold)
    print(json.dumps({"status":"generated","raw_candidates":len(candidates),"orbits":{k:len(sum(v.values(),[])) for k,v in splits.items()}}))

if __name__ == "__main__":
    main()

