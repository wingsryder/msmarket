#!/usr/bin/env python3
"""
Test script to fetch 1 year of data for all configured Indian NSE symbols and cache locally.
Behavior:
- Collect the full set of symbols from utils.config (all sectors under the INDIA market)
- Try a single batch fetch via yfinance.download for 1 year
- For any symbol missing/empty from the batch, try per-symbol fetch via Ticker(...).history(period='1y')
- If per-symbol still fails and a local cache file exists for that symbol, load the cached file as a fallback
- Save successful results per-symbol as CSV.gz in data/yf_cache/<SYMBOL>.csv.gz
- Prints a summary at the end (downloaded, per-symbol fallback, used cache fallback, failed)

Usage:
  python scripts/yf_cache_build.py [--output-dir data/yf_cache] [--period 1y]

Note: This is a standalone utility for testing and caching purposes. It does not
modify application logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import yfinance as yf

# Ensure we can import from repo root when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import MARKETS  # noqa: E402

REQUIRED_COLS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def get_all_symbols_from_config() -> List[str]:
    sectors = MARKETS["INDIA"]["sectors"]
    symbols: List[str] = []
    for _, syms in sectors.items():
        symbols.extend(syms)
    # Dedupe while preserving order
    seen = set()
    ordered: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def standardize_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Ensure the DataFrame has the REQUIRED_COLS with correct names and types."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    # Reset index to get date column if needed
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # Normalize the date/time column name to 'Date'
    if "Date" not in df.columns:
        # Heuristic: first column is often the datetime index after reset
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Date"})

    # Ensure required price columns exist; if missing, create empty
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Assign symbol and order/select columns
    df["Symbol"] = symbol

    # Parse Date to datetime if possible
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass

    return df[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]]


def save_symbol_cache(df: pd.DataFrame, symbol: str, output_dir: Path) -> bool:
    if df is None or df.empty:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip")
    return True


def load_symbol_cache(symbol: str, output_dir: Path) -> Optional[pd.DataFrame]:
    p = output_dir / f"{symbol}.csv.gz"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, compression="gzip")
        # best-effort date parse
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
            except Exception:
                pass
        return df
    except Exception:
        return None


def batch_download(symbols: List[str], period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            symbols,
            period=period,
            interval="1d",
            progress=False,
            threads=False,
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            repair=True,
        )
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def extract_symbol_from_batch(batch_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Given a batch df (possibly MultiIndex columns), extract one symbol tidy df."""
    if batch_df is None or batch_df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    # MultiIndex case: (symbol, field)
    if isinstance(batch_df.columns, pd.MultiIndex):
        if symbol not in batch_df.columns.get_level_values(0):
            return pd.DataFrame(columns=REQUIRED_COLS)
        sub = batch_df[symbol].copy()
        # move index to column
        sub = sub.reset_index()
        if "Date" not in sub.columns:
            sub = sub.rename(columns={sub.columns[0]: "Date"})
        sub = standardize_df(sub, symbol)
        return sub

    # Single symbol case
    sub = batch_df.reset_index()
    if "Date" not in sub.columns:
        sub = sub.rename(columns={sub.columns[0]: "Date"})
    sub = standardize_df(sub, symbol)
    return sub


def fetch_per_symbol(symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        tkr = yf.Ticker(symbol)
        df = tkr.history(period=period, interval="1d")
        if df is None or df.empty:
            return pd.DataFrame(columns=REQUIRED_COLS)
        df = standardize_df(df, symbol)
        return df
    except Exception:
        return pd.DataFrame(columns=REQUIRED_COLS)


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache 1y data for configured NSE symbols")
    parser.add_argument("--output-dir", default="data/yf_cache", help="Directory to store per-symbol CSV.gz files")
    parser.add_argument("--period", default="1y", help="yfinance period to request (default: 1y)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    symbols = get_all_symbols_from_config()
    print(f"Total symbols: {len(symbols)}")

    # First attempt: batch download for all symbols
    print("Attempting batch download via yfinance.download...")
    batch_df = batch_download(symbols, period=args.period)

    counts = {
        "downloaded": 0,
        "fallback_per_symbol": 0,
        "used_cache": 0,
        "failed": 0,
    }

    for sym in symbols:
        # Try to extract from batch first
        sym_df = extract_symbol_from_batch(batch_df, sym) if batch_df is not None else pd.DataFrame(columns=REQUIRED_COLS)

        # If batch is empty or missing this symbol, fallback to per-symbol fetch
        if sym_df.empty:
            per_df = fetch_per_symbol(sym, period=args.period)
            if not per_df.empty:
                sym_df = per_df
                counts["fallback_per_symbol"] += 1

        # If still empty, try to use cache
        if sym_df.empty:
            cached = load_symbol_cache(sym, output_dir)
            if cached is not None and not cached.empty:
                sym_df = standardize_df(cached, sym)
                counts["used_cache"] += 1

        if sym_df.empty:
            counts["failed"] += 1
            print(f"[FAIL] {sym}: no data from batch, per-symbol, or cache")
            continue

        # Save/overwrite cache file
        save_symbol_cache(sym_df, sym, output_dir)
        if counts["fallback_per_symbol"] == 0 and batch_df is not None:
            # Many likely came from batch; better to increment downloaded when we know
            counts["downloaded"] += 1
        else:
            # If we got here via batch for this symbol (not per-symbol and not cache)
            if load_symbol_cache(sym, output_dir) is not None:
                # Already saved. Count as downloaded if not counted as per-symbol or cache
                pass

        print(f"[OK] {sym}: rows={len(sym_df)}")

    print("\nSummary:")
    print(f"  Downloaded from batch (approx): {counts['downloaded']}")
    print(f"  Fallback via per-symbol:        {counts['fallback_per_symbol']}")
    print(f"  Used existing cache:            {counts['used_cache']}")
    print(f"  Failed:                         {counts['failed']}")
    print(f"  Cache directory:                {output_dir}")


if __name__ == "__main__":
    main()

