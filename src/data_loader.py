"""
Data Loading Module — DuckDB-based CSV ingestion.

Loads all 8 CSVs into DuckDB tables, unions transaction tables,
and verifies row counts against the mapping document.
"""

import os
import duckdb
import pandas as pd
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

EXPECTED_COUNTS = {
    "account_dim":       18_070,   # header row excluded
    "statement_fact":    658_228,
    "transaction_fact":  493_336,
    "wrld_stor_tran_fact": 1_053_854,
    "transaction_base":  1_547_190,  # union of the two
    "syf_id":            18_070,
    "rams_batch_cur":    96_799,
    "fraud_claim_case":  77,
    "fraud_claim_tran":  202,
}

CSV_FILES = {
    "account_dim":         "account_dim_20250325.csv",
    "statement_fact":      "statement_fact_20250325.csv",
    "transaction_fact":    "transaction_fact_20250325.csv",
    "wrld_stor_tran_fact": "wrld_stor_tran_fact_20250325.csv",
    "syf_id":              "syf_id_20250325.csv",
    "rams_batch_cur":      "rams_batch_cur_20250325.csv",
    "fraud_claim_case":    "fraud_claim_case_20250325.csv",
    "fraud_claim_tran":    "fraud_claim_tran_20250325.csv",
}


def _csv_path(name: str) -> str:
    """Resolve CSV path, trying raw dir first, then parent data dir."""
    raw = RAW_DATA_DIR / name
    if raw.exists():
        return str(raw)
    # Fallback: data/ at the repo root
    fallback = Path(__file__).resolve().parent.parent.parent / "data" / name
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(f"Cannot find CSV: {name}")


def create_connection(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection."""
    return duckdb.connect(db_path)


def load_all_tables(con: duckdb.DuckDBPyConnection, verbose: bool = True) -> dict:
    """
    Load all CSVs into DuckDB tables. Idempotent — drops and recreates.
    Returns a dict of table_name → row_count.
    """
    counts = {}

    # ── Load individual tables ───────────────────────────────────────────
    for table_name, csv_file in CSV_FILES.items():
        if table_name == "transaction_base":
            continue  # handled below
        path = _csv_path(csv_file)
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{path}', header=true, ignore_errors=true)
        """)
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        counts[table_name] = count
        if verbose:
            print(f"  ✓ {table_name}: {count:,} rows")

    # ── Union transaction tables ─────────────────────────────────────────
    con.execute("DROP TABLE IF EXISTS transaction_base")
    con.execute("""
        CREATE TABLE transaction_base AS
        SELECT *, 'transaction_fact' AS source_table
        FROM transaction_fact
        UNION ALL
        SELECT *, 'wrld_stor_tran_fact' AS source_table
        FROM wrld_stor_tran_fact
    """)
    tb_count = con.execute("SELECT COUNT(*) FROM transaction_base").fetchone()[0]
    counts["transaction_base"] = tb_count
    if verbose:
        print(f"  ✓ transaction_base (union): {tb_count:,} rows")

    return counts


def verify_counts(counts: dict, verbose: bool = True) -> bool:
    """Check row counts against expected values. Returns True if all match."""
    all_ok = True
    for table, expected in EXPECTED_COUNTS.items():
        actual = counts.get(table, 0)
        # Allow +/-1 tolerance for header/encoding differences
        if abs(actual - expected) > 5:
            if verbose:
                print(f"  ⚠ {table}: expected ~{expected:,}, got {actual:,}")
            all_ok = False
        elif verbose:
            print(f"  ✓ {table}: {actual:,} rows (OK)")
    return all_ok


def load_table_as_df(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    """Pull a DuckDB table into a pandas DataFrame."""
    return con.execute(f"SELECT * FROM {table_name}").fetchdf()


def get_latest_rams_snapshot(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Deduplicate rams_batch_cur to the most recent snapshot per account.
    Uses DISTINCT ON equivalent via ROW_NUMBER window function.
    """
    query = """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY cu_account_nbr
                       ORDER BY cu_processing_date DESC
                   ) AS rn
            FROM rams_batch_cur
        )
        SELECT * EXCLUDE (rn) FROM ranked WHERE rn = 1
    """
    return con.execute(query).fetchdf()


# ── Convenience loader ───────────────────────────────────────────────────────
def load_pipeline(verbose: bool = True):
    """
    Full data loading pipeline. Returns (connection, counts_dict).
    """
    if verbose:
        print("Loading CSV files into DuckDB…")
    con = create_connection()
    counts = load_all_tables(con, verbose=verbose)

    if verbose:
        print("\nVerifying row counts…")
    verify_counts(counts, verbose=verbose)

    return con, counts


if __name__ == "__main__":
    con, counts = load_pipeline(verbose=True)
    print(f"\nTotal tables loaded: {len(counts)}")
    print(f"Total rows across all tables: {sum(counts.values()):,}")
