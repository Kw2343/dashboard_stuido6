import pandas as pd
from pathlib import Path

d = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard_data")
cbf   = pd.read_csv(d / "cbf_dashboard.csv")
split = pd.read_csv(d / "ab_split_log.csv")

group_b = split[split["group"].str.contains("CBF") & (split["eligible"] == True)]
print(f"Group B users: {len(group_b)}")

for _, row in group_b.head(5).iterrows():
    uid      = row["user_id"]
    withheld = row["withheld_item"]
    user_cbf = cbf[cbf["user_id"] == uid]["parent_asin"].tolist()
    print(f"\nUser: {uid[:25]}")
    print(f"  Withheld item : {withheld}")
    print(f"  CBF recs      : {user_cbf}")
    print(f"  Hit           : {withheld in user_cbf}")