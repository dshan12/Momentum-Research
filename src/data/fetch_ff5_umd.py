import os
import io
import re
import zipfile
import requests
import pandas as pd
from io import StringIO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
OUT_PATH = os.path.join(ROOT, "data", "cleaned", "ff5_umd_monthly.csv")

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
UMD_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

YYYYMM_RE = re.compile(r"^\s*(\d{6})\b")


def _read_csv_block_from_zip(
    url: str, expected_header_hints: list[str]
) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = next(
        (n for n in z.namelist() if n.lower().endswith(".csv")), z.namelist()[0]
    )
    with z.open(name) as f:
        text = f.read().decode("latin-1")

    lines = text.splitlines()

    try:
        start = next(i for i, ln in enumerate(lines) if YYYYMM_RE.match(ln))
    except StopIteration:
        raise RuntimeError("Could not find start of monthly table (YYYYMM).")

    header_idx = None
    scan_above = range(max(0, start - 10), start)[::-1]
    for i in scan_above:
        ln_low = lines[i].lower()
        if any(hint.lower() in ln_low for hint in expected_header_hints):
            header_idx = i
            break
    if header_idx is None:
        header_idx = start - 1

    header_line = lines[header_idx]
    headers = [h.strip() for h in header_line.split(",")]

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if not lines[j].strip():
            end = j
            break
        if lines[j].lower().startswith(("annual", "annually", "yearly")):
            end = j
            break

    block = "\n".join([",".join(headers)] + lines[start:end])

    df = pd.read_csv(StringIO(block))
    first = df.columns[0]
    df = df.rename(columns={first: "yyyymm"})
    df["yyyymm"] = df["yyyymm"].astype(str).str.extract(r"(\d{6})", expand=False)
    df = df[df["yyyymm"].notna()].copy()
    df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df.drop(columns=["yyyymm"], inplace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main():
    ff5 = _read_csv_block_from_zip(
        FF5_URL, expected_header_hints=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    )
    keep_ff5 = [
        c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"] if c in ff5.columns
    ]
    if not keep_ff5:
        raise RuntimeError(
            f"FF5 expected columns not found; got: {ff5.columns.tolist()}"
        )
    ff5 = ff5[["date"] + keep_ff5]

    umd = _read_csv_block_from_zip(
        UMD_URL,
        expected_header_hints=["UMD", "Mom"],
    )
    mom_candidates = [
        c for c in umd.columns if c.strip().lower().startswith(("umd", "mom"))
    ]
    if not mom_candidates:
        raise RuntimeError(f"Momentum column not found; got: {umd.columns.tolist()}")
    umd = umd.rename(columns={mom_candidates[0]: "UMD"})[["date", "UMD"]]

    f = ff5.merge(umd, on="date", how="inner").sort_values("date")
    for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"]:
        if c in f.columns:
            f[c] = f[c] / 100.0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    f.to_csv(OUT_PATH, index=False)
    print(f"[✓] Saved {OUT_PATH} shape={f.shape}")


if __name__ == "__main__":
    main()
