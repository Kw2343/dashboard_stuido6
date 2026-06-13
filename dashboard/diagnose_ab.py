import pandas as pd
from pathlib import Path
 
d         = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard_data")
rev_path  = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\data\reviews_clean_no_exact_duplicates.csv")
 
cbf   = pd.read_csv(d / "cbf_dashboard.csv")
split = pd.read_csv(d / "ab_split_log.csv")
 
print("Split log columns:", list(split.columns))
print(f"Group B users: {len(split[split['group'].str.contains('CBF')])}")
 
# Load reviews to derive withheld item ourselves if column missing
reviews = pd.read_csv(rev_path, usecols=["user_id", "parent_asin", "timestamp_unix"],
                      low_memory=False)
 
group_b = split[split["group"].str.contains("CBF") & (split["eligible"] == True)]
 
hits_found = 0
for _, row in group_b.head(10).iterrows():
    uid = row["user_id"]
 
    # Get withheld item — from split log if column exists, else derive from reviews
    if "withheld_item" in split.columns:
        withheld = row["withheld_item"]
    else:
        ur = reviews[reviews["user_id"] == uid].sort_values("timestamp_unix")
        withheld = ur.iloc[-1]["parent_asin"] if len(ur) > 0 else ""
 
    user_cbf = cbf[cbf["user_id"] == uid]["parent_asin"].tolist()
    hit      = withheld in user_cbf
 
    if hit:
        hits_found += 1
 
    print(f"\nUser : {uid[:30]}")
    print(f"  Withheld item  : {withheld}")
    print(f"  CBF recs ({len(user_cbf):2}) : {user_cbf[:5]}{'...' if len(user_cbf) > 5 else ''}")
    print(f"  Hit            : {'✅ YES' if hit else '❌ NO'}")
 
print(f"\n{'='*50}")
print(f"Hits in first 10 Group B users: {hits_found}/10")
if hits_found == 0:
    print("\n⚠ CBF recs never match the withheld item.")
    print("  This means the CBF model recommends products")
    print("  SIMILAR to what the user bought, but not the")
    print("  exact withheld product itself.")
    print("  → The CBF only has 10 recs per user — too narrow.")
    print("  → Consider increasing top-N in CBF output, or")
    print("    evaluate using a broader relevance definition.")
 