"""Offline acceptance harness. Never import this module into the learner."""
from collections import deque
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import subprocess
import sys
from fixture_core import *

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def close(a,b,tol=1e-10):
    require(abs(float(a)-float(b))<=tol, f"{a} != {b}")

def expect_error(call, message):
    try:
        call()
    except ValueError as exc:
        require(str(exc)==message, f"Expected {message}, got {exc}")
    else:
        raise AssertionError("Expected error: "+message)

def main():
    topologies=read_json(ROOT/"data/topologies.json")
    expected=read_json(ROOT/"harness/expected_results.json")
    splits=read_json(ROOT/"data/split_manifest.json")
    checks=[]
    states=list(feasible_states())
    require(len(states)==924,"Feasible enumeration cardinality")
    for topo in topologies:
        name=topo["topology_id"]
        edges={tuple(p) for p in topo["pairs_distance_1"]}
        perms={tuple(p) for p in topo["symmetry_pullback_permutations"]}
        require(len(perms)==24,"Symmetry group order")
        for p in perms:
            require(sorted(p)==list(range(12)),"Invalid site permutation")
            require({tuple(sorted((p[i],p[j]))) for i,j in edges}==edges,"Symmetry changes the graph")
            for q in perms:
                require(tuple(p[q[i]] for i in range(12)) in perms,"Symmetries not closed")
        orbit_set={name+":"+canonical(topo,b) for b in states}
        # Independent expected counts, not inferred only from the generated gold file.
        require(len(orbit_set)==(50 if name=="ring12" else 55),"Wrong symmetry quotient")
        groups=[set(splits[name][key]) for key in ("training","development","holdout")]
        require(not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2],"Orbit leakage")
        require(set.union(*groups)==orbit_set,"Missing orbit in partition")
        for b in states:
            validate_bits(b)
            f=features(topo,b)
            for p in perms:
                require(features(topo,tuple(b[j] for j in p))==f,"Feature symmetry violation")
        # An XY exchange along graph edges must reach every fixed-count bit string.
        seen={states[0]}
        queue=deque(seen)
        while queue:
            b=queue.popleft()
            for i,j in edges:
                if b[i]!=b[j]:
                    moved=list(b); moved[i],moved[j]=moved[j],moved[i]
                    moved=tuple(moved)
                    if moved not in seen:
                        seen.add(moved); queue.append(moved)
        require(len(seen)==924,"Disconnected fixed-count swap graph")
        checks.append(name+": symmetry, grouped splits, constraints, swap reachability")
    # Verify datasets and the independent alternating-state lower bound.
    for case in ("core_pairwise","higher_order"):
        training=read_jsonl(ROOT/("data/"+case+"_training.jsonl"))
        development=read_jsonl(ROOT/("data/"+case+"_development.jsonl"))
        holdout=read_jsonl(ROOT/("harness/"+case+"_holdout.jsonl"))
        all_labels=read_jsonl(ROOT/("harness/"+case+"_all_labels.jsonl"))
        require([len(training),len(development),len(holdout),len(all_labels)]==[60,20,25,1848],"Dataset cardinality")
        check_energy_compatibility(training+development+holdout)
        for subset,key in ((training,"training"),(development,"development"),(holdout,"holdout")):
            for name in splits:
                rows=[r for r in subset if r["topology_id"]==name]
                require({r["orbit_id"] for r in rows}==set(splits[name][key]),"Split file mismatch")
                require(len(rows)==len({r["orbit_id"] for r in rows}),"Duplicate training orbit")
        for topo in topologies:
            name=topo["topology_id"]
            rows=[r for r in all_labels if r["topology_id"]==name]
            baseline=Fraction(3,25) if name=="ring12" else Fraction(1,5)
            lower=baseline-Fraction(len(topo["pairs_distance_1"]),50)
            close(min(r["energy_eV"] for r in rows),lower)
            close(lower,-.12 if name=="ring12" else -.16)
            # Unlike count cannot exceed edge count, and all other terms are nonnegative.
            alternating=tuple(map(int,"010101010101" if name=="ring12" else "010101101010"))
            f=features(topo,alternating)
            require(f[1]==len(topo["pairs_distance_1"]) and f[2:]==[0,0],"Lower bound not attained")
            for row in rows:
                f=features(topo,row["bits"])
                val=baseline-Fraction(f[1],50)+Fraction(3*f[2],500)
                if case=="higher_order": val+=Fraction(3*f[3],200)
                close(row["energy_eV"],val)
        for mode in ("none","constant","topology_dependent"):
            output=ROOT/"reference_outputs"/(case+"_"+mode+".json")
            subprocess.run([sys.executable,str(ROOT/"tools/reference_run.py"),"--case",case,"--correction",mode,"--out",str(output)],
                           check=True,stdout=subprocess.PIPE,text=True)
            report=read_json(output)
            truth=expected[case]["correction_tests"][mode]
            require(report["winning_topology"]==truth["winning_topology"],"Wrong topology winner")
            require(report["pooled_winner"]==report["winning_topology"],"Pool missed the winner")
            for topo in topologies:
                name=topo["topology_id"]
                result=report["topologies"][name]
                require(result["selected_model"]==expected[case]["expected_selected_model"],"Wrong selected model")
                close(result["minimum_energy_eV"],truth["minima_eV"][name])
                require(len(result["ground_state_orbits"])==1,"Wrong minimum-orbit count")
                selected=next(r for r in result["model_candidates"] if r["model"]==result["selected_model"])
                coef=list(map(Fraction,selected["coefficients_exact"]))
                errors=[predict(topo,r["bits"],selected["indices"],coef)-Fraction(str(r["energy_eV"]))
                        for r in holdout if r["topology_id"]==name]
                close(rmse(errors),0)
                if case=="higher_order":
                    expect_error(lambda:compile_pair_qubo(topo,coef),"UNSUPPORTED_HIGHER_ORDER_WITHOUT_QUADRATIZATION")
                elif mode=="none":
                    # Check all 4096 states, including invalid ratios, to catch penalty/sign/double-count errors.
                    qubo=result["qubo"]
                    scored=[]
                    for bits in product((0,1),repeat=12):
                        direct=predict(topo,bits,selected["indices"],coef)+(sum(bits)-6)**2
                        q=evaluate_qubo(qubo,bits)
                        close(q,direct)
                        scored.append((q,bits))
                    qmin=min(e for e,b in scored)
                    require(all(sum(b)==6 for e,b in scored if abs(e-qmin)<1e-10),"Penalty allowed invalid minimum")
                    close(qmin,truth["minima_eV"][name])
            checks.append(case+" / "+mode+": selection, untouched holdout, pool and exhaustive minima")
    invalid=read_json(ROOT/"harness/negative_inputs.json")
    for row in invalid["occupation_cases"]:
        expect_error(lambda:validate_bits(row["bits"]),row["expected_error"])
    sample=read_jsonl(ROOT/"data/core_pairwise_training.jsonl")[0]
    for mutation in invalid["energy_mutations"]:
        changed=deepcopy(sample); changed[mutation["field"]]=mutation["value"]
        expect_error(lambda:check_energy_compatibility([sample,changed]),mutation["expected_error"])
    expect_error(lambda:check_energy_compatibility([]),"EMPTY_ENERGY_SET")
    for row in invalid["normalization_cases"]:
        close(normalize_cell_energy(row["total_energy_eV"],row["n_formula_units"]),row["expected_normalized_eV"])
    for units in (0,-1,1.5,True):
        expect_error(lambda:normalize_cell_energy(-.1,units),"INVALID_FORMULA_UNIT_COUNT")
    checks.append("Negative inputs, fidelity separation, failed jobs, unit/normalization checks")
    summary={"status":"PASS","scope":"Fixture and bundled reference only; not a test of the user's pipeline", "check_groups":checks,
             "datasets":2,"correction_modes_per_dataset":3,"legal_states_per_graph":924,"qubo_states_checked":8192,
             "runtime_llm_calls":0,"real_DFT_calls":0,"quantum_circuits_run":0,
             "not_tested":["atomic construction","geometric screening","real potential accuracy","DFT","embedding","quantum hardware","iterative active learning","live literature review"]}
    write_json(ROOT/"reference_outputs/verification.json",summary)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
