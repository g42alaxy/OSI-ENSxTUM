"""Executable deterministic reference; no LLM, DFT, network or QPU calls."""
import argparse
import json
from pathlib import Path
from fixture_core import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=["core_pairwise","higher_order"],default="core_pairwise")
    parser.add_argument("--correction",choices=["none","constant","topology_dependent"],default="none")
    parser.add_argument("--out",default=None)
    args=parser.parse_args()
    topologies=read_json(ROOT/"data/topologies.json")
    models=read_json(ROOT/"data/model_candidates.json")["candidates"]
    training=read_jsonl(ROOT/("data/"+args.case+"_training.jsonl"))
    development=read_jsonl(ROOT/("data/"+args.case+"_development.jsonl"))
    check_energy_compatibility(training+development)
    for row in training+development:
        validate_bits(row["bits"])
    report={"provenance":"SYNTHETIC","runtime_agents_enabled":False,"case":args.case,"correction":args.correction,
            "note":"Reference algorithm only; no atomic construction, DFT, embedding or quantum circuit is executed.","topologies":{}}
    for topo in topologies:
        name=topo["topology_id"]
        tr=[r for r in training if r["topology_id"]==name]
        dv=[r for r in development if r["topology_id"]==name]
        results=[]
        for model in models:
            inds=model["feature_indices"]
            x=[[features(topo,r["bits"])[i] for i in inds] for r in tr]
            try:
                coef=fit_linear(x,[r["energy_eV"] for r in tr])
            except ValueError as e:
                results.append({"model":model["id"],"status":str(e)})
                continue
            errors=[predict(topo,r["bits"],inds,coef)-Fraction(str(r["energy_eV"])) for r in dv]
            results.append({"model":model["id"],"status":"OK","n_parameters":len(coef),"indices":inds,
                            "coefficients_exact":[str(c) for c in coef],"development_rmse_eV":rmse(errors)})
        passing=[r for r in results if r["status"]=="OK" and r["development_rmse_eV"]<=1e-10]
        if not passing:
            raise RuntimeError("MODEL_INADEQUATE: "+name)
        selected=min(passing,key=lambda r:(r["n_parameters"],r["model"]))
        coefficients=list(map(Fraction,selected["coefficients_exact"]))
        by_orbit={}
        for b in feasible_states():
            key=name+":"+canonical(topo,b)
            base=predict(topo,b,selected["indices"],coefficients)
            row={"orbit_id":key,"bits":list(b),"base_energy_eV":float(base),
                 "corrected_energy_eV":float(base+correction(name,b,args.correction,topo))}
            if key not in by_orbit or bit_string(b)<bit_string(by_orbit[key]["bits"]):
                by_orbit[key]=row
        ordered=sorted(by_orbit.values(),key=lambda r:(r["base_energy_eV"],r["orbit_id"]))
        cutoff=ordered[min(4,len(ordered)-1)]["base_energy_eV"]
        selected_pool=[r for r in ordered if r["base_energy_eV"]<=cutoff+1e-12]
        scored=[(predict(topo,b,selected["indices"],coefficients)+correction(name,b,args.correction,topo),b) for b in feasible_states()]
        minimum=min(e for e,b in scored)
        best=[b for e,b in scored if e==minimum]
        result={"model_candidates":results,"selected_model":selected["model"],"n_feasible":len(scored),
                "minimum_energy_eV":float(minimum),"minimizing_bitstrings":sorted(bit_string(b) for b in best),
                "ground_state_orbits":sorted({name+":"+canonical(topo,b) for b in best}),
                "base_model_encoding":"QUBO" if len(coefficients)==3 else "HUBO",
                "correction_application":"post-solve synthetic score; not compiled into base QUBO",
                "minimum_scope":"exhaustive corrected reference over all legal states",
                "n_symmetry_orbits":len(by_orbit),"selected_pool":selected_pool}
        if len(coefficients)==3 and args.correction=="none":
            qubo=compile_pair_qubo(topo,coefficients)
            result["qubo"]=qubo
        report["topologies"][name]=result
    report["ranked_pool"]=sorted([dict(row,topology_id=name) for name,res in report["topologies"].items() for row in res["selected_pool"]],
                                key=lambda r:(r["corrected_energy_eV"],r["orbit_id"]))
    report["pooled_winner"]=report["ranked_pool"][0]["topology_id"]
    report["winning_topology"]=min(report["topologies"],key=lambda name:report["topologies"][name]["minimum_energy_eV"])
    destination=Path(args.out) if args.out else ROOT/"reference_outputs"/(args.case+"_"+args.correction+".json")
    write_json(destination,report)
    print(json.dumps({"output":str(destination),"winner":report["winning_topology"],
                      "minima_eV":{n:r["minimum_energy_eV"] for n,r in report["topologies"].items()},
                      "selected_models":{n:r["selected_model"] for n,r in report["topologies"].items()}}))
if __name__=="__main__":
    main()

