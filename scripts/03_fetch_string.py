"""Fetch EZH2 protein-protein interactions and pathway enrichment from STRING v12.

Homo sapiens taxonomy ID: 9606. High-confidence interactions only (score > 700).

Outputs
-------
data/string/ezh2_ppi_network.json
data/string/ezh2_ppi_edges.csv
data/string/ezh2_pathway_enrichment.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, STRING_BASE, ensure_dirs, http_get

TAXON = 9606
CALLER = "ezh2_brca_research"
HIGH_CONFIDENCE = 700


def fetch_network():
    out_dir = os.path.join(DATA, "string")
    ensure_dirs(out_dir)

    partners = http_get(f"{STRING_BASE}/json/interaction_partners", params={
        "identifiers": "EZH2", "species": TAXON, "limit": 50,
        "caller_identity": CALLER, "network_type": "functional"})
    high_conf = [p for p in partners if p["score"] >= HIGH_CONFIDENCE / 1000]
    high_conf.sort(key=lambda p: -p["score"])
    top10 = high_conf[:10]

    core_names = [p["preferredName_B"] for p in top10]
    for canon in ("DNMT3A", "RBBP7"):
        if canon not in core_names:
            core_names.append(canon)
    partner_ids = ["EZH2"] + core_names
    network = http_get(f"{STRING_BASE}/json/network", params={
        "identifiers": "%0d".join(partner_ids), "species": TAXON,
        "caller_identity": CALLER, "network_type": "functional",
        "required_score": HIGH_CONFIDENCE})

    with open(os.path.join(out_dir, "ezh2_ppi_network.json"), "w") as fh:
        import json
        json.dump({"partners_all": partners, "network_core": network}, fh, indent=1)

    edges_path = os.path.join(out_dir, "ezh2_ppi_edges.csv")
    with open(edges_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["from", "to", "score", "score_normalized"])
        for e in network:
            writer.writerow([e["preferredName_A"], e["preferredName_B"],
                             e["score"], round(e["score"] / 1000, 3)])

    print(f"interaction partners: {len(partners)} total, {len(high_conf)} high-confidence")
    print("top 10:", [(p["preferredName_B"], round(p["score"], 3)) for p in top10])
    return network, top10


def fetch_enrichment():
    out_dir = os.path.join(DATA, "string")
    genes = ["EZH2", "EED", "SUZ12", "DNMT3A", "DNMT3B", "DNMT1", "RBBP4", "RBBP7",
             "AEBP2", "JARID2", "CBX2", "CBX4", "CBX7", "PHC1", "PHC2", "BMI1", "KMT2D", "BRCA1"]
    enrich = http_get(f"{STRING_BASE}/json/enrichment", params={
        "identifiers": "%0d".join(genes), "species": TAXON, "caller_identity": CALLER})
    rows = []
    for term in enrich:
        rows.append({
            "category": term["category"],
            "term_id": term["term"],
            "description": term["description"],
            "number_of_genes": term["number_of_genes"],
            "gene_names": term["preferredNames"],
            "FDR": term["fdr"],
            "neg_log10_FDR": round(-__import__("math").log10(term["fdr"]) if term["fdr"] > 0 else 0, 3),
        })
    rows.sort(key=lambda r: r["FDR"])
    path = os.path.join(out_dir, "ezh2_pathway_enrichment.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"enrichment terms: {len(rows)}")
    for r in rows[:8]:
        print(" ", r["category"], r["description"], "FDR=", r["FDR"])
    return path, rows


if __name__ == "__main__":
    fetch_network()
    fetch_enrichment()
    print("done")
