"""Parquet storage (one file per table per match) with DuckDB views on top."""

from pathlib import Path

import duckdb
import polars as pl

from cs2coach import config


def table_dir(table: str) -> Path:
    return config.PARQUET_DIR / table


def has_match(match_id: str) -> bool:
    return (table_dir("matches") / f"{match_id}.parquet").exists()


def write_match(match_id: str, frames: dict[str, pl.DataFrame]) -> None:
    for table, df in frames.items():
        d = table_dir(table)
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f"{match_id}.parquet")


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB file and (re)create one view per table over its parquet files."""
    config.ensure_dirs()
    con = duckdb.connect(str(config.DB_PATH), read_only=read_only)
    if not read_only:
        refresh_views(con)
    return con


def refresh_views(con: duckdb.DuckDBPyConnection) -> None:
    for table in config.TABLES:
        files = list(table_dir(table).glob("*.parquet"))
        if files:
            glob = str(table_dir(table) / "*.parquet")
            con.execute(
                f"CREATE OR REPLACE VIEW {table} AS "
                f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
            )


def load(table: str, where: str | None = None) -> pl.DataFrame:
    """Read a table straight from parquet with an optional SQL filter."""
    glob = str(table_dir(table) / "*.parquet")
    if not list(table_dir(table).glob("*.parquet")):
        return pl.DataFrame()
    sql = f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
    if where:
        sql += f" WHERE {where}"
    return duckdb.sql(sql).pl()
