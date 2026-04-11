"""Unit tests for clinical Cloud vs On-Prem concordance."""

import pandas as pd

from clinical_concordance import (
    cells_equal_normalized,
    default_strict_columns,
    numeric_within_tolerance,
    run_pairwise_concordance,
)


def test_cells_equal_normalized():
    assert cells_equal_normalized("a", "a")
    assert cells_equal_normalized("NA", "")
    assert cells_equal_normalized(None, "nan")
    assert not cells_equal_normalized("a", "b")


def test_numeric_within_tolerance():
    assert numeric_within_tolerance(10, 10, 10)
    assert numeric_within_tolerance(10, 10.5, 10)  # 10.5 in [9, 11]
    assert not numeric_within_tolerance(10, 12, 10)


def test_default_strict_columns():
    cols = ["FOO", "CDNA_CHG", "AA_CHG", "BAR"]
    d = default_strict_columns(cols)
    assert "CDNA_CHG" in d
    assert "AA_CHG" in d


def test_run_pairwise_concordance_perfect_match():
    rows = {
        "CHROM": ["chr1", "chr1"],
        "START": [100, 200],
        "REF": ["A", "G"],
        "ALT": ["G", "T"],
        "ENS_TRANS_ID": ["ENST1", "ENST2"],
        "GENE_NAME": ["G1", "G1"],
        "VARIANT_LOCATION": ["ONTARGET", "ONTARGET"],
        "CDNA_CHG": ["c.1A>G", "c.2G>T"],
        "SCORE": [10.0, 20.0],
    }
    df1 = pd.DataFrame(rows)
    df2 = pd.DataFrame(rows)
    r = run_pairwise_concordance(
        df1, df2, gene="G1", gene_col="GENE_NAME", strict_columns=["CDNA_CHG"],
        tolerance_pct=10.0,
        restrict_ontarget=True,
    )
    assert r.get("error") is None
    assert r["n_rows_matched"] == 2
    assert r["n_failed_strict"] == 0
    assert r["concordance_pct"] == 100.0


def test_run_pairwise_concordance_compare_all_genes():
    rows = {
        "CHROM": ["chr1", "chr1", "chr2"],
        "START": [100, 200, 300],
        "REF": ["A", "G", "C"],
        "ALT": ["G", "T", "T"],
        "ENS_TRANS_ID": ["ENST1", "ENST2", "ENST3"],
        "GENE_NAME": ["G1", "G2", "G1"],
        "VARIANT_LOCATION": ["ONTARGET", "ONTARGET", "ONTARGET"],
        "CDNA_CHG": ["a", "b", "c"],
        "SCORE": [10.0, 20.0, 30.0],
    }
    df1 = pd.DataFrame(rows)
    df2 = pd.DataFrame(rows)
    r = run_pairwise_concordance(
        df1,
        df2,
        compare_all_genes=True,
        gene_col="GENE_NAME",
        strict_columns=["CDNA_CHG"],
        tolerance_pct=10.0,
        restrict_ontarget=True,
    )
    assert r.get("error") is None
    assert r["n_rows_matched"] == 3
    assert r["concordance_pct"] == 100.0


def test_strict_mismatch():
    base = {
        "CHROM": ["chr1"],
        "START": [100],
        "REF": ["A"],
        "ALT": ["G"],
        "ENS_TRANS_ID": ["ENST1"],
        "GENE_NAME": ["G1"],
        "VARIANT_LOCATION": ["ONTARGET"],
        "CDNA_CHG": ["c.1A>G"],
        "SCORE": [10.0],
    }
    df1 = pd.DataFrame(base)
    b2 = dict(base)
    b2["CDNA_CHG"] = ["c.999Z"]
    df2 = pd.DataFrame(b2)
    r = run_pairwise_concordance(
        df1, df2, gene="G1", gene_col="GENE_NAME", strict_columns=["CDNA_CHG"],
        tolerance_pct=10.0,
    )
    assert r["n_failed_strict"] >= 1
    assert r["concordance_pct"] < 100.0
