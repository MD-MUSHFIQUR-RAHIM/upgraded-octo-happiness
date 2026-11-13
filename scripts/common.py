"""Shared constants and helpers for the EZH2-BRCA project."""
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(PROJECT, "data")
FIGURES = os.path.join(PROJECT, "figures")

GENE = "EZH2"
ENTREZ_ID = 2146
ENSEMBL_ID = "ENSG00000106462"
STUDY = "brca_tcga"
CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
XENA_HUB = "https://gdc.xenahubs.net"
XENA_EXPR_DATASET = "TCGA-BRCA.star_tpm.tsv"
XENA_CLINICAL_DATASET = "TCGA-BRCA.clinical.tsv"
STRING_BASE = "https://string-db.org/api"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "Mozilla/5.0 (research; EZH2-BRCA-student-project)"


def ensure_dirs(*subdirs):
    for d in subdirs:
        os.makedirs(d, exist_ok=True)


def http_get(url, params=None, retries=4, timeout=60, as_json=True):
    headers = {"User-Agent": USER_AGENT}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json() if as_json else r.text
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def http_post_json(url, payload, retries=4, timeout=120):
    headers = {"User-Agent": USER_AGENT}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            detail = getattr(exc, "response", None)
            if detail is not None and detail.text[:200]:
                print(f"  error detail: {detail.text[:200]}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {retries} attempts: {last}")
