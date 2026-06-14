#!/usr/bin/env python3
r"""
current_probe_02_source_data_audit.py

Current paper Probe 02
======================

Purpose
-------
Audit the paper's Nature/source-data XLSX files before running heavy model code.

This probe treats the source-data workbooks as the paper's plotted scoreboard.
It parses every .xlsx file in a source-data folder, exports raw sheets to CSV,
summarizes numeric columns, creates lightweight plots, and maps each workbook
to the paper claim it supports.

Current paper:
  "Scalable Boltzmann generators for equilibrium sampling of large-scale materials"
  Schebek, Noe, Rogal, Nature Communications (2026) 17:5010
  DOI: 10.1038/s41467-026-73900-9

Why this probe exists
---------------------
Before training or rerunning the official BG pipeline, we want reference rails:

  Figure2.xlsx -> RDF / energy histogram targets
  Figure3.xlsx -> ESS-vs-training targets
  Figure4.xlsx -> Helmholtz free-energy targets
  Figure5.xlsx -> Gibbs free-energy / cutoff / density targets
  Figure6.xlsx -> SW lambda3 phase-diagram targets
  Figure7.xlsx -> silicon phase-diagram targets
  SupplementaryFigure*.xlsx -> supporting diagnostics

This script deliberately avoids pandas/openpyxl and reads .xlsx files directly
using Python's standard library zip/xml stack. That makes it a robust first
pass in messy environments.

Expected usage
--------------
Expected repository layout:

  <repo-root>/
    41467_2026_73900_MOESM3_ESM/
      Figure2.xlsx
      Figure3.xlsx
      Figure4.xlsx
      Figure5.xlsx
      Figure6.xlsx
      Figure7.xlsx
      SupplementaryFigure1.xlsx
      SupplementaryFigure2.xlsx
      SupplementaryFigure3.xlsx
      SupplementaryFigure4.xlsx
      SupplementaryFigure5.xlsx
    probes/
      current_probe_02_source_data_audit.py

Run from the probes directory:

  cd .\probes
  python .\current_probe_02_source_data_audit.py

Or run from the repository root:

  python .\probes\current_probe_02_source_data_audit.py

Or point at a different source-data folder:

  python .\probes\current_probe_02_source_data_audit.py --source-dir path\to\41467_2026_73900_MOESM3_ESM

Optional:

  ... --no-plots
  ... --max-plot-series 20
  ... --dump-json-cells

Outputs
-------
A timestamped folder under probe_runs/, containing:

  PROBE02_SUMMARY.md
  source_data_inventory.csv
  workbook_sheet_summary.csv
  workbook_column_summary.csv
  figure_claim_targets.md
  figure_claim_targets.json
  sheet_csv/*.csv
  plots/*.png
  parsed_workbooks.json          optional summary
  parsed_cells.json              optional full-ish cells, if --dump-json-cells

Probe contract
--------------
This probe does not validate that the official code reproduces the figures.
It only establishes what is actually present in the source-data XLSX files and
turns those files into machine-readable targets for later probes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import re
import statistics
import sys
import textwrap
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


SCRIPT_DIR = Path(__file__).resolve().parent

# Probe files are normally stored under <repo-root>/probes.
# Resolve default paths relative to <repo-root>, not relative to probes/.
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "probes" else SCRIPT_DIR

DEFAULT_SOURCE_DIR = "41467_2026_73900_MOESM3_ESM"
EXPECTED_MAIN_FIGURES = [f"Figure{i}.xlsx" for i in range(2, 8)]
EXPECTED_SUPPLEMENTARY_FIGURES = [f"SupplementaryFigure{i}.xlsx" for i in range(1, 6)]
EXPECTED_WORKBOOKS = EXPECTED_MAIN_FIGURES + EXPECTED_SUPPLEMENTARY_FIGURES


# -----------------------------------------------------------------------------
# Optional plotting
# -----------------------------------------------------------------------------

def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Generic utils
# -----------------------------------------------------------------------------

def resolve_from_root(p: "str | Path") -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        seen = set()
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def sanitize_filename(name: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("._")
    if not s:
        s = "sheet"
    return s[:max_len]


def excel_col_to_index(col: str) -> int:
    idx = 0
    for ch in col.upper():
        if not ("A" <= ch <= "Z"):
            break
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def excel_ref_to_rc(ref: str) -> Tuple[int, int]:
    m = re.match(r"([A-Za-z]+)([0-9]+)", ref)
    if not m:
        raise ValueError(f"Bad cell ref: {ref!r}")
    c = excel_col_to_index(m.group(1))
    r = int(m.group(2)) - 1
    return r, c


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def coerce_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            return float(x)
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # Ignore common text labels.
        if s.lower() in {"nan", "none", "null", "true", "false"}:
            return None
        # Remove thousands separators but leave scientific notation.
        s2 = s.replace(",", "")
        try:
            v = float(s2)
            if math.isfinite(v):
                return v
        except Exception:
            return None
    return None


def value_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v) < 1e-14:
            v = 0.0
        return f"{v:.15g}"
    return str(v)


# -----------------------------------------------------------------------------
# XLSX parsing via stdlib zip/xml
# -----------------------------------------------------------------------------

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NS_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


@dataclass
class ParsedSheet:
    workbook: str
    sheet_name: str
    sheet_path: str
    n_rows: int
    n_cols: int
    rows: List[List[Any]]
    header_row_index: Optional[int]
    headers: List[str]
    data_start_index: int


@dataclass
class ParsedWorkbook:
    path: str
    filename: str
    sheets: List[ParsedSheet]


def _read_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    with zf.open(name) as f:
        return ET.fromstring(f.read())


def _safe_zip_name(base: str, target: str) -> str:
    # Excel relationship targets are often relative.
    if target.startswith("/"):
        return target.lstrip("/")
    base_dir = str(Path(base).parent).replace("\\", "/")
    combined = f"{base_dir}/{target}" if base_dir != "." else target
    parts: List[str] = []
    for part in combined.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = _read_xml(zf, "xl/sharedStrings.xml")
    out: List[str] = []
    for si in root.findall(f"{NS_MAIN}si"):
        # Shared string can be plain t or rich-text runs.
        texts = []
        for t in si.iter(f"{NS_MAIN}t"):
            texts.append(t.text or "")
        out.append("".join(texts))
    return out


def read_workbook_sheets(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    wb_path = "xl/workbook.xml"
    rels_path = "xl/_rels/workbook.xml.rels"
    wb = _read_xml(zf, wb_path)
    rels = _read_xml(zf, rels_path)

    rid_to_target: Dict[str, str] = {}
    for rel in rels.findall(f"{NS_PKG_REL}Relationship"):
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rid and target:
            rid_to_target[rid] = _safe_zip_name(wb_path, target)

    sheets = []
    for sh in wb.findall(f".//{NS_MAIN}sheet"):
        name = sh.attrib.get("name", "Sheet")
        rid = sh.attrib.get(f"{NS_REL}id")
        if rid and rid in rid_to_target:
            sheets.append((name, rid_to_target[rid]))
    return sheets


def parse_cell_value(c: ET.Element, shared_strings: Sequence[str]) -> Any:
    cell_type = c.attrib.get("t")
    v_elem = c.find(f"{NS_MAIN}v")
    is_elem = c.find(f"{NS_MAIN}is")

    if cell_type == "inlineStr":
        if is_elem is None:
            return ""
        texts = [t.text or "" for t in is_elem.iter(f"{NS_MAIN}t")]
        return "".join(texts)

    if v_elem is None:
        # Formula-only or blank cell; no cached value.
        return ""
    raw = v_elem.text or ""

    if cell_type == "s":
        try:
            idx = int(raw)
            return shared_strings[idx] if 0 <= idx < len(shared_strings) else raw
        except Exception:
            return raw
    if cell_type == "b":
        return raw == "1"
    if cell_type in {"str", "e"}:
        return raw

    # Default numeric.
    try:
        f = float(raw)
        if f.is_integer() and abs(f) < 9_007_199_254_740_992:
            return int(f)
        return f
    except Exception:
        return raw


def parse_sheet(zf: zipfile.ZipFile, workbook_filename: str, sheet_name: str, sheet_path: str, shared_strings: Sequence[str]) -> ParsedSheet:
    root = _read_xml(zf, sheet_path)
    cells: Dict[Tuple[int, int], Any] = {}
    max_r = 0
    max_c = 0

    for row in root.findall(f".//{NS_MAIN}sheetData/{NS_MAIN}row"):
        r_attr = row.attrib.get("r")
        row_idx_guess = int(r_attr) - 1 if r_attr and r_attr.isdigit() else max_r
        for c in row.findall(f"{NS_MAIN}c"):
            ref = c.attrib.get("r")
            if ref:
                r, col = excel_ref_to_rc(ref)
            else:
                r, col = row_idx_guess, max_c
            val = parse_cell_value(c, shared_strings)
            cells[(r, col)] = val
            max_r = max(max_r, r + 1)
            max_c = max(max_c, col + 1)

    rows: List[List[Any]] = []
    for r in range(max_r):
        rows.append([cells.get((r, c), "") for c in range(max_c)])

    header_idx, headers, data_start = infer_header(rows)
    return ParsedSheet(
        workbook=workbook_filename,
        sheet_name=sheet_name,
        sheet_path=sheet_path,
        n_rows=max_r,
        n_cols=max_c,
        rows=rows,
        header_row_index=header_idx,
        headers=headers,
        data_start_index=data_start,
    )


def parse_workbook(path: Path) -> ParsedWorkbook:
    with zipfile.ZipFile(path, "r") as zf:
        shared_strings = read_shared_strings(zf)
        sheet_refs = read_workbook_sheets(zf)
        sheets: List[ParsedSheet] = []
        for sheet_name, sheet_path in sheet_refs:
            if sheet_path not in zf.namelist():
                continue
            sheets.append(parse_sheet(zf, path.name, sheet_name, sheet_path, shared_strings))
    return ParsedWorkbook(path=str(path), filename=path.name, sheets=sheets)


# -----------------------------------------------------------------------------
# Header inference and sheet summaries
# -----------------------------------------------------------------------------

def infer_header(rows: List[List[Any]]) -> Tuple[Optional[int], List[str], int]:
    if not rows:
        return None, [], 0

    n_cols = max((len(r) for r in rows), default=0)
    if n_cols == 0:
        return None, [], 0

    # Find a row where following rows become numeric in at least two columns.
    best: Tuple[float, int] = (-1.0, 0)
    scan_limit = min(35, len(rows))
    for r in range(scan_limit):
        row = rows[r]
        nonempty = sum(1 for v in row if value_to_str(v).strip())
        strings = sum(1 for v in row if isinstance(v, str) and v.strip())
        nums_in_next = 0
        cols_with_nums = set()
        for rr in range(r + 1, min(len(rows), r + 12)):
            for c, v in enumerate(rows[rr]):
                if coerce_float(v) is not None:
                    nums_in_next += 1
                    cols_with_nums.add(c)
        # Penalize rows that are just one title cell, reward header-like text and numeric data below.
        score = 0.0
        if len(cols_with_nums) >= 2:
            score += len(cols_with_nums) * 2.0
            score += min(nums_in_next, 50) * 0.05
            score += min(nonempty, 20) * 0.3
            score += min(strings, 20) * 0.4
            if nonempty <= 1:
                score -= 2.0
        if score > best[0]:
            best = (score, r)

    if best[0] <= 0:
        # Fallback: no obvious header, use Excel-style column names.
        headers = [f"col_{i+1}" for i in range(n_cols)]
        return None, headers, 0

    header_idx = best[1]
    raw_headers = rows[header_idx]
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for c in range(n_cols):
        h = value_to_str(raw_headers[c] if c < len(raw_headers) else "").strip()
        if not h:
            h = f"col_{c+1}"
        # Flatten annoying newlines.
        h = re.sub(r"\s+", " ", h)
        base = h
        if base in seen:
            seen[base] += 1
            h = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        headers.append(h)

    return header_idx, headers, header_idx + 1


def sheet_to_records(sheet: ParsedSheet) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    headers = sheet.headers or [f"col_{i+1}" for i in range(sheet.n_cols)]
    for r in range(sheet.data_start_index, sheet.n_rows):
        row = sheet.rows[r]
        if not any(value_to_str(v).strip() for v in row):
            continue
        rec: Dict[str, Any] = {"__row_index_1based": r + 1}
        for c, h in enumerate(headers):
            rec[h] = row[c] if c < len(row) else ""
        records.append(rec)
    return records


def column_values(sheet: ParsedSheet, col_idx: int) -> List[Any]:
    vals = []
    for r in range(sheet.data_start_index, sheet.n_rows):
        row = sheet.rows[r]
        vals.append(row[col_idx] if col_idx < len(row) else "")
    return vals


def summarize_column(sheet: ParsedSheet, col_idx: int) -> Dict[str, Any]:
    header = sheet.headers[col_idx] if col_idx < len(sheet.headers) else f"col_{col_idx+1}"
    vals = column_values(sheet, col_idx)
    nonempty = [v for v in vals if value_to_str(v).strip()]
    nums = [coerce_float(v) for v in vals]
    nums2 = [v for v in nums if v is not None]
    out: Dict[str, Any] = {
        "workbook": sheet.workbook,
        "sheet": sheet.sheet_name,
        "column_index_1based": col_idx + 1,
        "header": header,
        "n_nonempty": len(nonempty),
        "n_numeric": len(nums2),
        "numeric_fraction_nonempty": (len(nums2) / len(nonempty)) if nonempty else 0.0,
        "kind_guess": "numeric" if nums2 and len(nums2) >= max(3, 0.6 * len(nonempty)) else "text/mixed",
    }
    if nums2:
        out.update(
            min=min(nums2),
            max=max(nums2),
            mean=sum(nums2) / len(nums2),
            median=statistics.median(nums2),
        )
        if len(nums2) >= 2:
            try:
                out["stdev"] = statistics.stdev(nums2)
            except Exception:
                out["stdev"] = ""
    samples = []
    for v in nonempty[:6]:
        samples.append(value_to_str(v))
    out["sample_values"] = " | ".join(samples)
    return out


def monotonic_score(xs: List[float]) -> float:
    if len(xs) < 3:
        return 0.0
    inc = sum(1 for a, b in zip(xs, xs[1:]) if b >= a)
    dec = sum(1 for a, b in zip(xs, xs[1:]) if b <= a)
    return max(inc, dec) / max(1, len(xs) - 1)


def pick_x_column(sheet: ParsedSheet, col_summaries: List[Dict[str, Any]]) -> Optional[int]:
    numeric_cols = [i for i, s in enumerate(col_summaries) if s["kind_guess"] == "numeric" and s["n_numeric"] >= 3]
    if not numeric_cols:
        return None

    preferred_patterns = [
        r"\br\b", r"r/", r"distance", r"training", r"step", r"\bN\b", r"particle",
        r"lambda", r"λ", r"cut", r"radius", r"\bT\b", r"temperature", r"\bP\b", r"pressure",
    ]
    for i in numeric_cols:
        h = str(col_summaries[i]["header"])
        if any(re.search(p, h, flags=re.IGNORECASE) for p in preferred_patterns):
            return i

    # Prefer monotonic numeric column.
    best = (0.0, numeric_cols[0])
    for i in numeric_cols:
        vals = [coerce_float(v) for v in column_values(sheet, i)]
        xs = [v for v in vals if v is not None]
        score = monotonic_score(xs)
        if score > best[0]:
            best = (score, i)
    return best[1]


# -----------------------------------------------------------------------------
# Claim-target mapping
# -----------------------------------------------------------------------------

FIGURE_TARGETS: Dict[str, Dict[str, Any]] = {
    "Figure2.xlsx": {
        "paper_claim": "Large-system transferred local BG samples reproduce RDFs and potential-energy histograms close to MD/base comparisons, without reweighting.",
        "expected_objects": ["RDF g(r)", "energy histogram", "base", "BG", "MD", "FCC LJ N=1372", "cubic mW ice N=1728"],
        "later_probe_use": "Reference curves for official-output comparison after tiny/local reproduction.",
        "failure_watch": "RDF agreement may hide energy/weight/correlation failure.",
    },
    "Figure3.xlsx": {
        "paper_claim": "Local BGs reach higher effective sample size than global BGs as training progresses.",
        "expected_objects": ["ESS", "training steps", "local", "global", "mW N=64", "mW N=216", "FCC LJ N=256"],
        "later_probe_use": "Training-efficiency target and ESS curve sanity check.",
        "failure_watch": "ESS variance and convergence speed; local success may be potential/structure dependent.",
    },
    "Figure4.xlsx": {
        "paper_claim": "Size-transferred local BGs provide accurate Helmholtz free energies for cubic mW ice and cubic-vs-hexagonal free-energy differences.",
        "expected_objects": ["Helmholtz free energy", "particle number N", "joint", "marginal", "MD+MBAR", "cubic ice", "hexagonal ice"],
        "later_probe_use": "Free-energy target rails and joint-vs-marginal comparison.",
        "failure_watch": "Low ESS can still produce deceptively small variance; marginal estimates may correct joint bias.",
    },
    "Figure5.xlsx": {
        "paper_claim": "Shape-conditional local BGs estimate Gibbs free-energy differences and densities for FCC/HCP LJ across cutoff radii.",
        "expected_objects": ["Gibbs free energy", "cutoff radius", "density", "FCC", "HCP", "N=1080", "training cutoff"],
        "later_probe_use": "Cutoff/long-range sensitivity reference.",
        "failure_watch": "Changing cutoff radius is already a locality/long-range stress axis.",
    },
    "Figure6.xlsx": {
        "paper_claim": "Local BGs conditioned on Stillinger-Weber three-body strength lambda3 reproduce phase stability across structures and system sizes.",
        "expected_objects": ["lambda3", "Gibbs free energy", "diamond cubic", "beta-tin", "BCC", "training sizes", "large sizes"],
        "later_probe_use": "Parameter-conditioning and system-size transfer reference.",
        "failure_watch": "Subtle finite-size phase-boundary shifts require accurate long-range/collective statistics.",
    },
    "Figure7.xlsx": {
        "paper_claim": "Temperature/pressure-conditioned local BGs reproduce the silicon diamond-cubic vs beta-tin coexistence line.",
        "expected_objects": ["temperature", "pressure", "phase diagram", "Gibbs free-energy difference", "coexistence line", "silicon"],
        "later_probe_use": "Thermodynamic-conditioning target.",
        "failure_watch": "Phase boundary may be sensitive to structure-specific performance and marginal-density issues.",
    },
}


def target_for_file(filename: str) -> Dict[str, Any]:
    if filename in FIGURE_TARGETS:
        return dict(FIGURE_TARGETS[filename])
    if filename.lower().startswith("supplementaryfigure"):
        return {
            "paper_claim": "Supplementary source data supporting paper diagnostics.",
            "expected_objects": ["supplementary diagnostics"],
            "later_probe_use": "Inspect for additional ESS/free-energy/timing/M-sensitivity/correlation targets.",
            "failure_watch": "May contain the actual seam or caveat hidden behind main-figure summary.",
        }
    return {
        "paper_claim": "Unmapped workbook.",
        "expected_objects": [],
        "later_probe_use": "Manual inspection required.",
        "failure_watch": "Unknown.",
    }


def guess_sheet_category(workbook: str, sheet: ParsedSheet, col_summaries: List[Dict[str, Any]]) -> str:
    hay = " ".join(
        [workbook, sheet.sheet_name]
        + [str(s.get("header", "")) for s in col_summaries]
        + [str(s.get("sample_values", "")) for s in col_summaries[:10]]
    ).lower()

    categories = []
    checks = [
        ("rdf", ["rdf", "g(r)", "radial", "r/"]),
        ("energy_histogram", ["energy", "density", "hist"]),
        ("ess_training", ["ess", "effective sample", "training", "step"]),
        ("helmholtz_free_energy", ["helmholtz", "free energy", " f", "f/"]),
        ("gibbs_free_energy", ["gibbs", "delta g", "Δg", "dg", "cutoff"]),
        ("phase_diagram", ["phase", "lambda", "λ", "pressure", "temperature", "coexistence"]),
        ("density", ["density", "rho", "ρ"]),
        ("timing_cost", ["gpu", "time", "wall", "cost"]),
        ("marginal_joint", ["marginal", "joint", "m=200"]),
    ]
    for label, terms in checks:
        if any(t.lower() in hay for t in terms):
            categories.append(label)

    return ";".join(categories) if categories else "unclassified_numeric_source_data"


# -----------------------------------------------------------------------------
# Exporting and plotting
# -----------------------------------------------------------------------------

def export_sheet_csv(sheet: ParsedSheet, csv_dir: Path) -> Path:
    safe = sanitize_filename(f"{Path(sheet.workbook).stem}__{sheet.sheet_name}.csv")
    path = csv_dir / safe

    headers = sheet.headers or [f"col_{i+1}" for i in range(sheet.n_cols)]
    rows = sheet_to_records(sheet)
    fieldnames = ["__row_index_1based"] + headers
    write_csv(path, rows, fieldnames=fieldnames)
    return path


def make_sheet_plot(sheet: ParsedSheet, col_summaries: List[Dict[str, Any]], plot_dir: Path, max_series: int) -> Optional[Path]:
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    numeric_cols = [i for i, s in enumerate(col_summaries) if s["kind_guess"] == "numeric" and s["n_numeric"] >= 3]
    if len(numeric_cols) < 2:
        return None

    x_col = pick_x_column(sheet, col_summaries)
    if x_col is None:
        return None
    y_cols = [i for i in numeric_cols if i != x_col][:max_series]
    if not y_cols:
        return None

    x_vals_all = [coerce_float(v) for v in column_values(sheet, x_col)]
    x_header = sheet.headers[x_col] if x_col < len(sheet.headers) else f"col_{x_col+1}"

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    plotted = 0
    for y_col in y_cols:
        y_vals_all = [coerce_float(v) for v in column_values(sheet, y_col)]
        xs: List[float] = []
        ys: List[float] = []
        for x, y in zip(x_vals_all, y_vals_all):
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if len(xs) < 3:
            continue
        y_header = sheet.headers[y_col] if y_col < len(sheet.headers) else f"col_{y_col+1}"
        ax.plot(xs, ys, linewidth=1.2, label=str(y_header)[:80])
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title(f"{sheet.workbook} — {sheet.sheet_name}")
    ax.set_xlabel(str(x_header))
    ax.set_ylabel("numeric value")
    if plotted <= 12:
        ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()

    out = plot_dir / f"{sanitize_filename(Path(sheet.workbook).stem + '__' + sheet.sheet_name)}.png"
    ensure_dir(out.parent)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_claim_targets_md(path: Path, present_files: Sequence[str]) -> None:
    lines = [
        "# Current Probe 02 — figure/source-data claim targets",
        "",
        "This file maps each source-data workbook to the paper claim it can later test.",
        "Probe 02 only audits the workbooks; it does not reproduce the model.",
        "",
    ]
    for fname in sorted(present_files):
        target = target_for_file(fname)
        lines += [
            f"## {fname}",
            "",
            f"**Paper claim:** {target['paper_claim']}",
            "",
            "**Expected objects:**",
            "",
        ]
        for obj in target.get("expected_objects", []):
            lines.append(f"- {obj}")
        lines += [
            "",
            f"**Later probe use:** {target['later_probe_use']}",
            "",
            f"**Failure watch:** {target['failure_watch']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(
    path: Path,
    out_dir: Path,
    source_dir: Path,
    inventory: List[Dict[str, Any]],
    sheet_summary: List[Dict[str, Any]],
    column_summary: List[Dict[str, Any]],
    plot_count: int,
) -> None:
    found = len(inventory)
    total_sheets = len(sheet_summary)
    total_numeric_cols = sum(1 for r in column_summary if r.get("kind_guess") == "numeric")
    missing_expected = [f for f in EXPECTED_WORKBOOKS if f not in {r["filename"] for r in inventory}]

    # Top categories.
    cat_counts: Dict[str, int] = {}
    for row in sheet_summary:
        for cat in str(row.get("category_guess", "")).split(";"):
            if not cat:
                continue
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    lines = [
        "# PROBE 02 SUMMARY",
        "",
        f"Output folder: `{out_dir}`",
        f"Source-data folder: `{source_dir}`",
        "",
        "## What this probe did",
        "",
        "- Parsed every `.xlsx` workbook in the source-data folder using Python stdlib zip/xml.",
        "- Exported every parsed sheet to CSV.",
        "- Summarized workbook/sheet dimensions and numeric columns.",
        "- Generated lightweight plots for numeric sheets.",
        "- Mapped main-figure workbooks to the paper claims they support.",
        "",
        "## Headline counts",
        "",
        f"- Workbooks found: **{found}**",
        f"- Sheets parsed: **{total_sheets}**",
        f"- Numeric columns detected: **{total_numeric_cols}**",
        f"- Plots generated: **{plot_count}**",
        "",
    ]

    if missing_expected:
        lines += [
            "## Missing expected source-data workbooks",
            "",
        ]
        for f in missing_expected:
            lines.append(f"- `{f}`")
        lines.append("")
    else:
        lines += ["## Missing expected source-data workbooks", "", "_None._", ""]

    lines += [
        "## Category guesses",
        "",
    ]
    for cat, n in sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{cat}`: {n} sheet(s)")
    lines.append("")

    lines += [
        "## Workbooks",
        "",
    ]
    for row in inventory:
        lines.append(
            f"- `{row['filename']}`: {row['n_sheets']} sheet(s), "
            f"{row['total_rows']} row(s), {row['total_cols']} max col(s), "
            f"claim target: {row['claim_target_short']}"
        )
    lines.append("")

    lines += [
        "## Main outputs",
        "",
        "- `source_data_inventory.csv`",
        "- `workbook_sheet_summary.csv`",
        "- `workbook_column_summary.csv`",
        "- `figure_claim_targets.md`",
        "- `figure_claim_targets.json`",
        "- `sheet_csv/*.csv`",
        "- `plots/*.png`",
        "",
        "## Next probe",
        "",
        "Probe 03 should run the smallest official repo/tutorial path and compare whatever it emits against these source-data targets where possible.",
        "",
        "Probe 04+ should start the locality stress sequence: neighbor/cutoff sweeps, system-size transfer, then controlled long-range injection.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Current paper Probe 02: source-data XLSX audit.")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Folder containing the Nature source-data workbooks, relative to the repository root unless absolute.")
    p.add_argument("--out-root", default="probes/probe_runs", help="Output root directory, relative to the repository root unless absolute.")
    p.add_argument("--no-plots", action="store_true", help="Disable lightweight PNG plot generation.")
    p.add_argument("--max-plot-series", type=int, default=18, help="Maximum y-series per sheet plot.")
    p.add_argument("--dump-json-cells", action="store_true", help="Write parsed cell data to parsed_cells.json. Can be verbose.")
    p.add_argument("--strict-main-figures", action="store_true", help="Exit nonzero if Figure2.xlsx-Figure7.xlsx are not all present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve_from_root(args.source_dir)
    out_dir = ensure_dir(resolve_from_root(args.out_root) / f"current_probe02_source_data_audit_{now_stamp()}")
    csv_dir = ensure_dir(out_dir / "sheet_csv")
    plot_dir = ensure_dir(out_dir / "plots")

    print("=" * 100)
    print("CURRENT PAPER PROBE 02 — SOURCE-DATA XLSX AUDIT")
    print("=" * 100)
    print(f"[SOURCE] {source_dir}")
    print(f"[OUT]    {out_dir}")

    if not source_dir.exists():
        print(f"[ERROR] Source-data directory does not exist: {source_dir}", file=sys.stderr)
        print(f"Expected folder name: {DEFAULT_SOURCE_DIR}", file=sys.stderr)
        print("Expected files: " + ", ".join(EXPECTED_WORKBOOKS), file=sys.stderr)
        sys.exit(2)

    xlsx_files = sorted(p for p in source_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if not xlsx_files:
        print(f"[ERROR] No .xlsx files found in {source_dir}", file=sys.stderr)
        sys.exit(2)

    all_workbooks: List[ParsedWorkbook] = []
    inventory: List[Dict[str, Any]] = []
    sheet_summary: List[Dict[str, Any]] = []
    column_summary: List[Dict[str, Any]] = []
    plot_rows: List[Dict[str, Any]] = []
    full_cells: Dict[str, Any] = {}

    for path in xlsx_files:
        print(f"[XLSX] {path.name}")
        try:
            wb = parse_workbook(path)
        except Exception as exc:
            inventory.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "parse_ok": False,
                    "error": repr(exc),
                    "n_sheets": 0,
                    "total_rows": 0,
                    "total_cols": 0,
                    "claim_target_short": target_for_file(path.name)["paper_claim"][:120],
                }
            )
            print(f"  [WARN] parse failed: {exc!r}")
            continue

        all_workbooks.append(wb)
        inv = {
            "filename": path.name,
            "path": str(path),
            "parse_ok": True,
            "error": "",
            "n_sheets": len(wb.sheets),
            "total_rows": sum(s.n_rows for s in wb.sheets),
            "total_cols": max([s.n_cols for s in wb.sheets] or [0]),
            "claim_target_short": target_for_file(path.name)["paper_claim"][:160],
        }
        inventory.append(inv)

        if args.dump_json_cells:
            full_cells[path.name] = {
                "sheets": [
                    {
                        "sheet_name": s.sheet_name,
                        "n_rows": s.n_rows,
                        "n_cols": s.n_cols,
                        "header_row_index_0based": s.header_row_index,
                        "headers": s.headers,
                        "rows": s.rows,
                    }
                    for s in wb.sheets
                ]
            }

        for s in wb.sheets:
            records = sheet_to_records(s)
            csv_path = export_sheet_csv(s, csv_dir)
            col_summaries = [summarize_column(s, c) for c in range(s.n_cols)]
            category = guess_sheet_category(path.name, s, col_summaries)
            numeric_cols = [r for r in col_summaries if r["kind_guess"] == "numeric"]
            text_cols = [r for r in col_summaries if r["kind_guess"] != "numeric"]

            sheet_summary.append(
                {
                    "workbook": path.name,
                    "sheet": s.sheet_name,
                    "sheet_path": s.sheet_path,
                    "n_rows": s.n_rows,
                    "n_cols": s.n_cols,
                    "header_row_index_1based": "" if s.header_row_index is None else s.header_row_index + 1,
                    "data_start_row_1based": s.data_start_index + 1,
                    "n_data_records_nonempty": len(records),
                    "n_numeric_columns": len(numeric_cols),
                    "n_text_or_mixed_columns": len(text_cols),
                    "category_guess": category,
                    "exported_csv": str(csv_path),
                }
            )
            column_summary.extend(col_summaries)

            if not args.no_plots:
                plot_path = make_sheet_plot(s, col_summaries, plot_dir, max_series=max(1, args.max_plot_series))
                if plot_path:
                    plot_rows.append(
                        {
                            "workbook": path.name,
                            "sheet": s.sheet_name,
                            "plot": str(plot_path),
                        }
                    )

    target_json = {fname: target_for_file(fname) for fname in sorted([p.name for p in xlsx_files])}

    write_csv(out_dir / "source_data_inventory.csv", inventory)
    write_csv(out_dir / "workbook_sheet_summary.csv", sheet_summary)
    write_csv(out_dir / "workbook_column_summary.csv", column_summary)
    write_csv(out_dir / "generated_plots.csv", plot_rows)
    write_json(out_dir / "figure_claim_targets.json", target_json)
    write_claim_targets_md(out_dir / "figure_claim_targets.md", [p.name for p in xlsx_files])

    parsed_summary = {
        "source_dir": str(source_dir),
        "n_workbooks": len(inventory),
        "n_sheets": len(sheet_summary),
        "n_columns": len(column_summary),
        "workbooks": inventory,
        "sheets": sheet_summary,
        "plots": plot_rows,
    }
    write_json(out_dir / "parsed_workbooks.json", parsed_summary)
    if args.dump_json_cells:
        write_json(out_dir / "parsed_cells.json", full_cells)

    write_summary_md(
        out_dir / "PROBE02_SUMMARY.md",
        out_dir=out_dir,
        source_dir=source_dir,
        inventory=inventory,
        sheet_summary=sheet_summary,
        column_summary=column_summary,
        plot_count=len(plot_rows),
    )

    expected = set(EXPECTED_WORKBOOKS)
    present = {p.name for p in xlsx_files}
    missing = sorted(expected - present)
    missing_main = sorted(set(EXPECTED_MAIN_FIGURES) - present)
    missing_supplementary = sorted(set(EXPECTED_SUPPLEMENTARY_FIGURES) - present)

    print("\n[SUMMARY]")
    print(f"  workbooks_found       : {len(xlsx_files)}")
    print(f"  workbooks_parsed_ok   : {sum(1 for r in inventory if r['parse_ok'])}")
    print(f"  sheets_parsed         : {len(sheet_summary)}")
    print(f"  numeric_columns       : {sum(1 for r in column_summary if r.get('kind_guess') == 'numeric')}")
    print(f"  plots_generated       : {len(plot_rows)}")
    print(f"  missing_main_figures  : {missing_main if missing_main else 'none'}")
    print(f"  missing_supplementary : {missing_supplementary if missing_supplementary else 'none'}")
    print(f"  missing_expected_all  : {missing if missing else 'none'}")
    print(f"  output                : {out_dir}")

    print("\nNext:")
    print(f"  open {out_dir / 'PROBE02_SUMMARY.md'}")
    print(f"  open {out_dir / 'figure_claim_targets.md'}")
    print(f"  open {out_dir / 'workbook_sheet_summary.csv'}")

    if args.strict_main_figures and missing_main:
        sys.exit(3)


if __name__ == "__main__":
    main()
