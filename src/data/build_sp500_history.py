import re
import os
import typing as T
import datetime as dt
from dataclasses import dataclass
from collections import defaultdict
from io import StringIO

import pandas as pd
import requests

WIKI_CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_CHANGES_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#Changes_in_2020s"
)

YF_TICKER_FIX = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit /537.36"
            "(KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def normalize_ticker(t: str) -> str:
    if pd.isna(t):
        return None
    t = t.strip().upper()
    t = t.replace(" ", "")
    if t in YF_TICKER_FIX:
        return YF_TICKER_FIX[t]
    t = re.sub(r"\.(?=[A-Z0-9]+$)", "-", t)
    return t


def parse_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    try:
        return pd.to_datetime(s, errors="raise").normalize()
    except Exception:
        try:
            d = pd.to_datetime(s, errors="raise")
            return (d + pd.offsets.MonthEnd(0)).normalize()
        except Exception:
            return pd.NaT


@dataclass
class ChangeEvent:
    date: pd.Timestamp
    added: T.List[str]
    removed: T.List[str]
    raw: dict


def _extract_current_constituents() -> pd.DataFrame:
    html = fetch_html(WIKI_CURRENT_URL)
    df_list = pd.read_html(StringIO(html), flavor="bs4")
    candidates = [
        df
        for df in df_list
        if any(
            c.lower() in ["symbol", "ticker"]
            for c in [str(col).lower() for col in df.columns]
        )
    ]
    if not candidates:
        raise RuntimeError("Failed to extract current constituents")
    cur = candidates[0].copy()
    colmap = {c: c for c in cur.columns}
    for c in cur.columns:
        if str(c).strip().lower() in ["symbol", "ticker", "code"]:
            colmap[c] = "Symbol"
        if str(c).strip().lower() in ["security", "company", "name"]:
            colmap[c] = "Security"
        if "gics" in str(c).lower() and "sector" in str(c).lower():
            colmap[c] = "GICS Sector"
    cur = cur.rename(columns=colmap)
    cur["Symbol"] = cur["Symbol"].apply(normalize_ticker)
    cur = cur.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
    return cur[
        ["Symbol", "Security"]
        + [c for c in cur.columns if c not in ["Symbol", "Security"]]
    ]


def _extract_changes_table() -> pd.DataFrame:
    html = fetch_html(WIKI_CURRENT_URL)
    all_tables = pd.read_html(StringIO(html), flavor="bs4")

    cand = []
    for df in all_tables:
        cols_lower = [str(c).strip().lower() for c in df.columns]
        has_date = any("date" in c for c in cols_lower)
        has_changeish = any(
            any(
                k in c
                for k in [
                    "added",
                    "removed",
                    "company",
                    "ticker",
                    "reason",
                    "change",
                    "action",
                    "notes",
                ]
            )
            for c in cols_lower
        )
        if has_date and has_changeish:
            cand.append(df)

    if not cand:
        raise RuntimeError("Failed to extract changes table")

    frames = []
    for t in cand:
        df = t.copy()
        df.columns = [str(c).strip() for c in df.columns]

        std = {}
        taken = set()
        for c in list(df.columns):
            lc = str(c).lower()
            target = None
            if "date" in lc:
                target = "Date"
            elif "added" in lc:
                target = "Added"
            elif "removed" in lc:
                target = "Removed"
            elif "reason" in lc or "notes" in lc or "change" in lc or "action" in lc:
                target = "Reason"
            elif "symbol" in lc or "ticker" in lc:
                target = "Ticker"
            elif "company" in lc:
                target = "Company"
            if target and target not in taken:
                std[c] = target
                taken.add(target)

        if std:
            df = df.rename(columns=std)

        # ensure unique column labels (drops later duplicates keeping first)
        df = df.loc[:, ~df.columns.duplicated()]
        keep = [
            c
            for c in ["Date", "Added", "Removed", "Reason", "Ticker", "Company"]
            if c in df.columns
        ]
        if not keep or "Date" not in keep:
            continue

        # ensure Date is parsed
        df = df[keep].copy()
        if "Date" in df:
            df["Date"] = df["Date"].apply(parse_date)
            df = df.dropna(subset=["Date"])

        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("Failed to extract changes table")

    changes = (
        pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    )
    return changes


def _parse_added_removed(row: pd.Series) -> ChangeEvent:
    added, removed = list(), list()
    for col, bucket in [("Added", added), ("Removed", removed)]:
        if col in row and pd.notna(row[col]):
            s = str(row[col]).strip()
            parts = re.split(r"[,/]| and ", s)
            for p in parts:
                p = p.strip()
                if "Ticker" in row and pd.notna(row["Ticker"]):
                    pass
                if p:
                    m = re.search(r"\(([A-Za-z0-9\.\-]+)\)", p)
                    if m:
                        bucket.append(normalize_ticker(m.group(1)))
                    else:
                        bucket.append(normalize_ticker(p))
    if not added and not removed and "Reason" in row and pd.notna(row["Reason"]):
        txt = str(row["Reason"])
        m = re.search(
            r"([A-Za-z0-9\.\-]+)\s+replac(?:e|es|ing)\s+([A-Za-z0-9\.\-]+)", txt, re.I
        )
        if m:
            added = [normalize_ticker(m.group(1))]
            removed = [normalize_ticker(m.group(2))]

    added = [a for a in added if a]
    removed = [r for r in removed if r]
    return ChangeEvent(
        date=row["Date"], added=added, removed=removed, raw=row.to_dict()
    )


def build_membership_timeline(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a dataframe of daily membership from 'start' to 'end'
    Strategy:
        1. Currnet set from current page
        2. Changes_in_2020s All changes table since 1990
        3. Roll back form 'today' to 'start' by reversing each change
        4. Roll forward day-by-day applying changes
    """
    current = _extract_current_constituents()
    current_set = set(current["Symbol"].dropna().tolist())
    changes = _extract_changes_table()
    events: T.List[ChangeEvent] = []
    for _, row in changes.iterrows():
        ev = _parse_added_removed(row)
        if pd.isna(ev.date):
            continue
        if not ev.added and not ev.removed:
            continue
        events.append(ev)

    events = [e for e in events if e.date >= pd.Timestamp("1990-01-01")]
    events.sort(key=lambda e: e.date)

    today = pd.Timestamp(dt.date.today())
    backward = [e for e in events if e.date > start]
    backward.reverse()
    memb = set(current_set)
    for e in backward:
        for a in e.added:
            if a in memb:
                memb.remove(a)
        for r in e.removed:
            if r:
                memb.add(r)

    by_date: T.DefaultDict[pd.Timestamp, T.List[ChangeEvent]] = defaultdict(list)
    for e in events:
        if start <= e.date <= end:
            by_date[e.date.normalize()].append(e)
    days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    daily_rows = list()
    current_members = set(memb)

    for d in days:
        if d in by_date:
            for ev in by_date[d]:
                for r in ev.removed:
                    if r in current_members:
                        current_members.remove(r)
                for a in ev.added:
                    if a:
                        current_members.add(a)
        for t in current_members:
            daily_rows.append((d, t, 1))
    daily = pd.DataFrame(daily_rows, columns=["date", "ticker", "in_index"])
    return daily


def monthly_panel_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily["month"] = pd.to_datetime(daily["date"]) + pd.offsets.MonthEnd(0)
    monthly = (
        daily.groupby(["month", "ticker"])["in_index"]
        .max()
        .reset_index()
        .rename(columns={"month": "date"})
    )
    return monthly


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", default="data/cleaned/sp500_membership_monthly.csv")
    args = parser.parse_args()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    print("[*] Building survivorship-bias–free membership…")
    daily = build_membership_timeline(start, end)
    monthly = (
        monthly_panel_from_daily(daily)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    monthly["ticker"] = monthly["ticker"].apply(normalize_ticker)

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    monthly.to_csv(out, index=False)
    print(f"[✓] Wrote {out} with {len(monthly):,} rows.")


if __name__ == "__main__":
    main()
