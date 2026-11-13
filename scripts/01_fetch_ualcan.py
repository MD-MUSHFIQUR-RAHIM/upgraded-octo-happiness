"""Fetch EZH2 expression (tumor vs normal + PAM50 subtype) and survival data.

UALCAN serves TCGA data through a web front end without a documented public API,
so this script pulls the identical underlying TCGA data programmatically:

  * Expression: UCSC Xena GDC hub, TCGA-BRCA STAR TPM (log2(TPM+1)), which
    includes 1,106 tumor and 113 normal samples - the same quantification
    pipeline UALCAN displays.
  * PAM50 subtypes: UCSC Xena TCGA legacy hub BRCA_clinicalMatrix
    (PAM50_mRNA_nature2012, ER/PR/HER2_Status_nature2012).
  * Overall + disease-free survival: cBioPortal REST API (brca_tcga).

Outputs
-------
data/ualcan/ezh2_expression_tumor_vs_normal.csv
data/ualcan/ezh2_expression_by_subtype.csv
data/ualcan/ezh2_survival_data.csv
"""
import csv
import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CBIOPORTAL_BASE, DATA, GENE, STUDY, XENA_CLINICAL_DATASET,
                    XENA_EXPR_DATASET, XENA_HUB, ensure_dirs, http_get)

import requests
import xenaPython as xena

XENA_LEGACY_HUB = "https://tcga.xenahubs.net"
XENA_LEGACY_CLINICAL = "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"


def fetch_expression():
    ualcan_dir = os.path.join(DATA, "ualcan")
    ensure_dirs(ualcan_dir)

    samples = xena.dataset_samples(XENA_HUB, XENA_EXPR_DATASET, None)
    res = xena.dataset_gene_probes_values(XENA_HUB, XENA_EXPR_DATASET, samples, [GENE])
    values = res[1][0]

    expr_rows = []
    for sample, value in zip(samples, values):
        parts = sample.split("-")
        code = parts[3][:2] if len(parts) > 3 else ""
        sample_type = {"01": "Tumor", "11": "Normal", "06": "Metastatic"}.get(code, "Other")
        expr_rows.append({
            "sample_id": sample,
            "sample_type_code": code,
            "sample_type": sample_type,
            "EZH2_log2TPM": round(value, 4),
        })

    expr_path = os.path.join(ualcan_dir, "ezh2_expression_tumor_vs_normal.csv")
    pd.DataFrame(expr_rows).to_csv(expr_path, index=False)

    counts = {}
    for r in expr_rows:
        counts[r["sample_type"]] = counts.get(r["sample_type"], 0) + 1
    print(f"expression rows: {len(expr_rows)} {counts}")
    return expr_path, expr_rows


def fetch_legacy_pam50():
    url = f"{XENA_LEGACY_HUB}/download/{XENA_LEGACY_CLINICAL}"
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 (research)"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    cols = ["sampleID", "PAM50_mRNA_nature2012", "ER_Status_nature2012",
            "HER2_Final_Status_nature2012", "PR_Status_nature2012"]
    df = df[cols].copy()
    df["sample_id"] = df["sampleID"].astype(str)
    df["patient_id"] = df["sample_id"].str.split("-").str[:3].str.join("-")
    print(f"legacy clinical rows: {len(df)}, with PAM50: "
          f"{(df['PAM50_mRNA_nature2012'].notna()).sum()}")
    return df


def fetch_cbioportal_clinical():
    recs = http_get(f"{CBIOPORTAL_BASE}/studies/{STUDY}/clinical-data",
                    params={"clinicalDataType": "PATIENT", "projection": "DETAILED"})
    by_attr = {}
    patient_id_key = "patientId"
    for rec in recs:
        by_attr.setdefault(rec["clinicalAttributeId"], {})[rec[patient_id_key]] = rec["value"]
    print(f"cBioPortal OS records: {len(by_attr.get('OS_STATUS', {}))}, "
          f"DFS records: {len(by_attr.get('DFS_STATUS', {}))}")
    return by_attr


def build_tables(expr_rows, pam50_df, clinical):
    ualcan_dir = os.path.join(DATA, "ualcan")
    expr_df = pd.DataFrame(expr_rows)
    expr_df["patient_id"] = expr_df["sample_id"].str.split("-").str[:3].str.join("-")
    pam50_df = pam50_df.drop_duplicates("patient_id")

    merged = expr_df.merge(pam50_df[["patient_id", "PAM50_mRNA_nature2012",
                                     "ER_Status_nature2012", "HER2_Final_Status_nature2012",
                                     "PR_Status_nature2012"]],
                           on="patient_id", how="left")
    merged = merged.rename(columns={
        "PAM50_mRNA_nature2012": "PAM50_subtype",
        "ER_Status_nature2012": "ER_STATUS",
        "HER2_Final_Status_nature2012": "HER2_STATUS",
        "PR_Status_nature2012": "PR_STATUS",
    })

    def get(attr, patient):
        return clinical.get(attr, {}).get(patient, "")

    merged["OS_STATUS"] = merged["patient_id"].map(lambda p: get("OS_STATUS", p))
    merged["OS_MONTHS"] = merged["patient_id"].map(lambda p: to_float(get("OS_MONTHS", p)))
    merged["DFS_STATUS"] = merged["patient_id"].map(lambda p: get("DFS_STATUS", p))
    merged["DFS_MONTHS"] = merged["patient_id"].map(lambda p: to_float(get("DFS_MONTHS", p)))

    surv = merged[merged["sample_type"] == "Tumor"].copy()
    surv_path = os.path.join(ualcan_dir, "ezh2_survival_data.csv")
    surv.to_csv(surv_path, index=False)

    sub = surv[surv["PAM50_subtype"].notna() & (surv["PAM50_subtype"] != "NA")].copy()
    sub_path = os.path.join(ualcan_dir, "ezh2_expression_by_subtype.csv")
    sub.to_csv(sub_path, index=False)

    print(f"survival rows: {len(surv)} (OS present: {(surv['OS_STATUS'] != '').sum()}, "
          f"PAM50 subtype: {len(sub)})")
    print("subtypes:", sub["PAM50_subtype"].value_counts().to_dict())
    return surv_path, sub_path


def to_float(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    _, expr_rows = fetch_expression()
    pam50_df = fetch_legacy_pam50()
    clinical = fetch_cbioportal_clinical()
    build_tables(expr_rows, pam50_df, clinical)
    print("done")
