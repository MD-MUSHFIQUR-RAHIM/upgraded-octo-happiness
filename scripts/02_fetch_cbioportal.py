"""Fetch EZH2 mutation and copy-number data from the cBioPortal REST API.

Outputs
-------
data/cbioportal/ezh2_mutations.json
data/cbioportal/ezh2_cna.json
data/cbioportal/ezh2_mutation_summary.csv
data/cbioportal/ezh2_cna_summary.csv
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CBIOPORTAL_BASE, DATA, ENTREZ_ID, STUDY, ensure_dirs,
                    http_get, http_post_json)


def fetch_mutations():
    out_dir = os.path.join(DATA, "cbioportal")
    ensure_dirs(out_dir)
    mutations = http_get(
        f"{CBIOPORTAL_BASE}/molecular-profiles/{STUDY}_mutations/mutations",
        params={"sampleListId": f"{STUDY}_all", "entrezGeneId": ENTREZ_ID,
                "projection": "DETAILED"})
    with open(os.path.join(out_dir, "ezh2_mutations.json"), "w") as fh:
        import json
        json.dump(mutations, fh, indent=1)

    n_samples_total = http_get(f"{CBIOPORTAL_BASE}/sample-lists/{STUDY}_all")["sampleCount"]
    by_type = Counter(m["mutationType"] for m in mutations)
    by_protein = Counter(m.get("proteinChange", "NA") for m in mutations)
    n_mut_samples = len({m["sampleId"] for m in mutations})

    summary_rows = []
    for mtype, count in by_type.most_common():
        summary_rows.append({"category": mtype, "count": count,
                             "fraction_of_cohort": round(count / n_samples_total, 4)})
    summary_path = os.path.join(out_dir, "ezh2_mutation_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "count", "fraction_of_cohort"])
        writer.writeheader()
        writer.writerows(summary_rows)

    lollipop_path = os.path.join(out_dir, "ezh2_lollipop_positions.csv")
    with open(lollipop_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["protein_position", "protein_change", "mutation_type", "count"])
        pos_counter = Counter()
        for m in mutations:
            pos = m.get("proteinPosStart") or m.get("proteinPosEnd") or 0
            pos_counter[(pos, m.get("proteinChange", "NA"), m["mutationType"])] += 1
        for (pos, change, mtype), count in sorted(pos_counter.items()):
            writer.writerow([pos, change, mtype, count])

    print(f"mutations: {len(mutations)} in {n_mut_samples}/{n_samples_total} samples "
          f"({100 * n_mut_samples / n_samples_total:.1f}%)")
    print("types:", dict(by_type))
    print("top protein changes:", by_protein.most_common(8))
    return mutations, n_samples_total


def fetch_cna():
    out_dir = os.path.join(DATA, "cbioportal")
    cna = http_post_json(
        f"{CBIOPORTAL_BASE}/molecular-profiles/{STUDY}_gistic/discrete-copy-number/fetch",
        {"sampleListId": f"{STUDY}_all",
         "entrezGeneIds": [ENTREZ_ID]})
    with open(os.path.join(out_dir, "ezh2_cna.json"), "w") as fh:
        import json
        json.dump(cna, fh, indent=1)

    n_samples_total = http_get(f"{CBIOPORTAL_BASE}/sample-lists/{STUDY}_all")["sampleCount"]
    by_alteration = Counter(c["alteration"] for c in cna)
    labels = {-2: "Deep deletion", -1: "Shallow deletion", 0: "Diploid", 1: "Gain", 2: "Amplification"}
    rows = []
    for alt in (2, 1, 0, -1, -2):
        count = by_alteration.get(alt, 0)
        rows.append({"alteration": labels[alt], "count": count,
                     "fraction_of_cohort": round(count / n_samples_total, 4)})
    path = os.path.join(out_dir, "ezh2_cna_summary.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["alteration", "count", "fraction_of_cohort"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CNA records: {len(cna)} of {n_samples_total} samples")
    print({labels[k]: v for k, v in sorted(by_alteration.items(), reverse=True)})
    return cna


if __name__ == "__main__":
    fetch_mutations()
    fetch_cna()
    print("done")
