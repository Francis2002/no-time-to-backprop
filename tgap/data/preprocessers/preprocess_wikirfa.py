# preprocess_wikirfa.py
import gzip, io, re, numpy as np, pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List
from pathlib import Path

RFC2822 = "%H:%M, %d %B %Y"   # matches "19:53, 25 January 2013"

def _parse_blocks_from_gz(path: str) -> List[dict]:
    """Parse wiki-RfA.txt.gz into a list of dicts with keys: src, tgt, vot, res, yea, dat, txt."""
    rows = []
    with gzip.open(path, "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")
    # Split on blank lines
    blocks = re.split(r"\n\s*\n", content.strip())
    for b in blocks:
        d = {}
        for line in b.splitlines():
            if line.startswith("SRC:"):
                d["src"] = line[4:].strip()
            elif line.startswith("TGT:"):
                d["tgt"] = line[4:].strip()
            elif line.startswith("VOT:"):
                d["vot"] = int(line[4:].strip())
            elif line.startswith("RES:"):
                d["res"] = int(line[4:].strip())
            elif line.startswith("YEA:"):
                d["yea"] = int(line[4:].strip())
            elif line.startswith("DAT:"):
              try:
                # Try to parse the date to validate it
                date_str = line[4:].strip()
                # Additional validation for time components
                time_part = date_str.split(',')[0]  # Extract "HH:MM" part
                hour, minute = map(int, time_part.split(':'))
                if hour > 23 or minute > 59:
                  raise ValueError(f"Invalid time: {time_part}")
                d["dat"] = date_str
              except ValueError:
                # Invalid date format, skip this entry by not adding "dat" key
                pass
            elif line.startswith("TXT:"):
                d["txt"] = line[4:].strip()
        if "src" in d and "tgt" in d and "vot" in d and "dat" in d:
            rows.append(d)
    return rows

def _map_ids(str_ids: np.ndarray) -> Tuple[np.ndarray, Dict[str,int]]:
    uniq = np.unique(str_ids)
    m = {k: i for i, k in enumerate(uniq)}
    out = np.vectorize(lambda z: m[z])(str_ids)
    return out.astype(np.int32), m

def _time_sincos(ts: np.ndarray):
    hod = (ts % 86400) / 3600.0
    dow = ((ts // 86400) + 4) % 7
    return np.stack([
        np.sin(2*np.pi*hod/24.0),
        np.cos(2*np.pi*hod/24.0),
        np.sin(2*np.pi*dow/7.0),
        np.cos(2*np.pi*dow/7.0),
    ], axis=1).astype(np.float32)

def _dt_endpoints(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, num_nodes: int):
    last = np.full((num_nodes,), -1, dtype=np.int64)
    dt_s = np.empty_like(ts)
    dt_d = np.empty_like(ts)
    for i in range(ts.shape[0]):
        u, v, t = int(src[i]), int(dst[i]), int(ts[i])
        lu, lv = last[u], last[v]
        dt_s[i] = 0 if lu < 0 else max(0, t - lu)
        dt_d[i] = 0 if lv < 0 else max(0, t - lv)
        last[u] = t
        last[v] = t
    return np.log1p(dt_s).astype(np.float32), np.log1p(dt_d).astype(np.float32)

def preprocess_wikirfa_gz(
    gz_path: str,
    out_npz: str,
    *,
    drop_neutral: bool = False,   # True => binary ± (common in SOTA); False => 3-class {-1,0,+1}
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
):
    """
    Produces strictly-chronological stream:
      src:int32, dst:int32, feat:[dt_src, dt_dst, hod_sin, hod_cos, dow_sin, dow_cos], target, t:int64
    """

    # Check if output file already exists
    if Path(out_npz).exists():
        print(f"[*] Output file {out_npz} already exists. Using existing file.")
        return
    
    rows = _parse_blocks_from_gz(gz_path)

    # Parse timestamp (RFC2822-like; the page example matches "%H:%M, %d %B %Y")
    # Check for blank fields before parsing timestamps
    # Remove rows where one of the required fields "src", "tgt", "vot" is missing
    filtered_rows = []
    removed_count_src = 0
    removed_count_tgt = 0
    removed_count_vot = 0
    removed_count_dat = 0
    removed_count = 0
    for i, r in enumerate(rows):
      if all(key in r and str(r[key]).strip() for key in ["src", "tgt", "vot", "dat"]):
        filtered_rows.append(r)
      else:
        missing_fields = [key for key in ["src", "tgt", "vot", "dat"] if key not in r or not str(r[key]).strip()]
        if "src" in missing_fields:
          removed_count_src += 1
        if "tgt" in missing_fields:
          removed_count_tgt += 1
        if "vot" in missing_fields:
          removed_count_vot += 1
        if "dat" in missing_fields:
          removed_count_dat += 1
        removed_count += 1
    if removed_count_src > 0:
      print(f"[!] Removed {removed_count_src} rows with missing or blank 'src' field.")
    if removed_count_tgt > 0:
      print(f"[!] Removed {removed_count_tgt} rows with missing or blank 'tgt' field.")
    if removed_count_vot > 0:
      print(f"[!] Removed {removed_count_vot} rows with missing or blank 'vot' field.")
    if removed_count_dat > 0:
      print(f"[!] Removed {removed_count_dat} rows with missing or blank 'dat' field.")
    if removed_count > 0:
      print(f"[!] Removed a total of {removed_count} rows with missing or blank required fields.")
      print(f"[!] {removed_count_src + removed_count_tgt + removed_count_vot + removed_count_dat - removed_count} rows had multiple missing fields.")
    if len(filtered_rows) == 0:
      raise ValueError("No valid rows found after filtering. Please check the input file.")
    rows = filtered_rows
    
    def fix_date_typos(date_str):
      import re
      
      # Common typos in month names
      typo_fixes = {
          "Januaryuary": "January",
          "Janry": "January", 
          "Jan": "January",
          "Feb": "February",
          "Mar": "March",
          "Apr": "April",
          "Mya": "May",
          "Junee": "June",
          "Jun": "June", 
          "Julu": "July",
          "Jul": "July",
          "Aug": "August",
          "Sepember": "September",
          "Sep": "September",
          "Oct": "October",
          "Octoberober": "October", 
          "Novmber": "November",
          "Nov": "November",
          "Decmber": "December",
          "Dec": "December",
      }
      
      # Use word boundaries to avoid partial matches
      for typo, correction in typo_fixes.items():
          # \b ensures we only match whole words
          pattern = r'\b' + re.escape(typo) + r'\b'
          date_str = re.sub(pattern, correction, date_str, flags=re.IGNORECASE)
      
      return date_str
    
    ts = np.array([
      int(datetime.strptime(fix_date_typos(r["dat"]), RFC2822).timestamp()) for r in rows
    ], dtype=np.int64)

    src_raw = np.array([r["src"] for r in rows], dtype=object)
    dst_raw = np.array([r["tgt"] for r in rows], dtype=object)
    vot_raw = np.array([int(r["vot"]) for r in rows], dtype=np.int32)

    # Sort by time; stable mergesort retains input order for identical timestamps
    order = np.argsort(ts, kind="mergesort")
    ts, src_raw, dst_raw, vot_raw = ts[order], src_raw[order], dst_raw[order], vot_raw[order]

    # Map user strings to contiguous node ids
    all_ids = np.concatenate([src_raw, dst_raw])
    all_map, idmap = _map_ids(all_ids)
    num_nodes = len(idmap)
    src = all_map[:src_raw.shape[0]]
    dst = all_map[src_raw.shape[0]:]

    # Optional: drop neutrals for binary sign task (common in signed-link SOTA)
    if drop_neutral:
        keep = (vot_raw != 0)
        src, dst, ts, vot_raw = src[keep], dst[keep], ts[keep], vot_raw[keep]

    # Features: Δt per endpoint + calendar sin/cos (all causal)
    dt_s, dt_d = _dt_endpoints(src, dst, ts, num_nodes)
    cal = _time_sincos(ts)
    feat = np.concatenate([dt_s[:,None], dt_d[:,None], cal], axis=1).astype(np.float32)

    # Targets:
    if drop_neutral:
        # map {-1,0,+1} -> {-1,1}
        y = np.where(vot_raw > 0, 1, -1).astype(np.int32)[:, None]
    else:
        # keep {-1,0,+1} but shift to {0,1,2} for convenience (model can remap)
        y = (vot_raw + 1).astype(np.int32)[:, None]

    # Chronological split
    N = src.shape[0]
    i_tr = int(round((1.0 - val_ratio - test_ratio) * N))
    i_va = int(round((1.0 - test_ratio) * N))
    idx_train = np.arange(0, i_tr, dtype=np.int32)
    idx_val   = np.arange(i_tr, i_va, dtype=np.int32)
    idx_test  = np.arange(i_va, N, dtype=np.int32)

    np.savez_compressed(
        out_npz,
        src=src, dst=dst, feat=feat, target=y, t=ts,
        num_nodes=np.array([num_nodes], dtype=np.int32),
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        meta=np.array([b"wiki-rfa"], dtype=object),
    )
    print(f"[*] Saved {out_npz} | N={N} nodes={num_nodes} feat_dim={feat.shape[1]} drop_neutral={drop_neutral}")
