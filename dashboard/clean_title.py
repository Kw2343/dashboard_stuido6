#!/usr/bin/env python3
"""
update_clean_title.py
=====================
Replaces the `title` column in products_clean.csv with the
`cleaned_title` value from cleaned_titles.csv, matched on parent_asin.
 
- Products with no match in cleaned_titles keep their original title unchanged.
- Original file is backed up before saving.
"""
 
from pathlib import Path
import shutil
import pandas as pd
 
PRODUCTS_FILE       = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\data\products_clean.csv")
CLEANED_TITLES_FILE = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\data\cleaned_titles.csv")
BACKUP_FILE         = PRODUCTS_FILE.with_suffix(".backup.csv")
 
SEP = "─" * 60
 
def main():
    print(SEP)
    print("  Replace title with cleaned_title in products_clean.csv")
    print(SEP)
 
    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[LOAD]  {PRODUCTS_FILE.name}")
    if not PRODUCTS_FILE.exists():
        print(f"  ERROR: {PRODUCTS_FILE} not found"); return
    products = pd.read_csv(PRODUCTS_FILE, low_memory=False)
    products["parent_asin"] = products["parent_asin"].astype(str).str.strip()
    print(f"  {len(products):,} rows")
 
    print(f"[LOAD]  {CLEANED_TITLES_FILE.name}")
    if not CLEANED_TITLES_FILE.exists():
        print(f"  ERROR: {CLEANED_TITLES_FILE} not found"); return
    cleaned = pd.read_csv(CLEANED_TITLES_FILE, low_memory=False)
    cleaned["parent_asin"] = cleaned["parent_asin"].astype(str).str.strip()
    print(f"  {len(cleaned):,} rows")
 
    # ── Backup ────────────────────────────────────────────────────────────────
    shutil.copy2(PRODUCTS_FILE, BACKUP_FILE)
    print(f"\n[BACKUP] {BACKUP_FILE.name}")
 
    # ── Build lookup: parent_asin → cleaned_title ─────────────────────────────
    # Filter out invalid cleaned_title values (0, empty, whitespace-only)
    valid = cleaned[["parent_asin", "cleaned_title"]].copy()
    valid["cleaned_title"] = valid["cleaned_title"].astype(str).str.strip()
    valid = valid[
        valid["cleaned_title"].notna() &
        (valid["cleaned_title"] != "") &
        (valid["cleaned_title"] != "0") &
        (valid["cleaned_title"] != "nan")
    ]
    invalid_count = len(cleaned) - len(valid)
    if invalid_count > 0:
        print(f"[WARN]  Skipped {invalid_count:,} rows with invalid cleaned_title (0, empty, etc.)")
 
    lookup = (
        valid
        .drop_duplicates("parent_asin")
        .set_index("parent_asin")["cleaned_title"]
    )
 
    # ── Replace title where a cleaned_title exists ────────────────────────────
    original_titles = products["title"].copy()
    products["title"] = products["parent_asin"].map(lookup).fillna(products["title"])
 
    replaced  = (products["title"] != original_titles).sum()
    unchanged = len(products) - replaced
 
    print(f"\n[RESULT]")
    print(f"  Titles replaced  : {replaced:,}")
    print(f"  Titles unchanged : {unchanged:,}  (no match in cleaned_titles)")
    print(f"  Total rows       : {len(products):,}")
 
    # ── Sample ────────────────────────────────────────────────────────────────
    changed_mask = products["title"] != original_titles
    print(f"\n[SAMPLE] 5 replaced titles:")
    sample = products[changed_mask][["parent_asin", "title"]].head(5)
    sample.insert(2, "original_title", original_titles[changed_mask].head(5).values)
    print(sample.to_string(index=False))
 
    # ── Save ──────────────────────────────────────────────────────────────────
    products.to_csv(PRODUCTS_FILE, index=False)
    print(f"\n[SAVE]  {PRODUCTS_FILE}")
    print(f"\n{SEP}\n  Done.\n{SEP}")
 
if __name__ == "__main__":
    main()