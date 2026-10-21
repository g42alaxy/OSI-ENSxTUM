"""Explicit synthetic oracle boundary for integration tests; never use as physical evidence."""
import argparse
from fixture_core import ROOT, read_json, read_jsonl, write_jsonl, validate_bits, correction
from generate_fixture import make_record

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,help="JSONL requests with topology_id and bits")
    parser.add_argument("--out",required=True)
    parser.add_argument("--case",choices=["core_pairwise","higher_order"],default="core_pairwise")
    parser.add_argument("--correction",choices=["none","constant","topology_dependent"],default="none")
    args=parser.parse_args()
    topologies={t["topology_id"]:t for t in read_json(ROOT/"data/topologies.json")}
    responses=[]
    for request in read_jsonl(args.input):
        validate_bits(request["bits"])
        topo=topologies[request["topology_id"]]
        row=make_record(topo,request["bits"],args.case)
        row["synthetic_correction_eV"]=float(correction(topo["topology_id"],request["bits"],args.correction,topo))
        row["corrected_energy_eV"]=row["energy_eV"]+row["synthetic_correction_eV"]
        row["corrected_fidelity"]="SYNTHETIC_CORRECTED_"+args.case.upper()+"_"+args.correction.upper()
        row["correction_mode"]=args.correction
        responses.append(row)
    write_jsonl(args.out,responses)
    print(f"Wrote {len(responses)} explicitly synthetic results to {args.out}")

if __name__=="__main__":
    main()
