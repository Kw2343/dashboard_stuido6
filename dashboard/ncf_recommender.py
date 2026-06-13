"""
ncf_recommender.py
==================
Neural Collaborative Filtering (NCF) Recommender System
Implements GMF + MLP fusion (He et al. 2017 — "Neural Collaborative Filtering")
 
OUTPUTS (saved to ncf_results/):
  1. ncf_training_log.csv        — epoch-level loss + metrics (plot the loss curve in your dashboard)
  2. ncf_user_recommendations.csv — top-N recs per user (same schema as CBF output)
  3. ncf_summary.csv             — model-level Hit Rate / Precision / Recall vs Popularity baseline
  4. ncf_evaluation_detail.csv   — per-user hit / precision / recall (deep dive)
 
REQUIREMENTS:
  pip install torch pandas numpy scikit-learn tqdm
 
USAGE:
  python ncf_recommender.py
"""
 
import warnings
warnings.filterwarnings("ignore")
 
import random
import math
from pathlib  import Path
from datetime import datetime
 
import numpy  as np
import pandas as pd
from tqdm import tqdm
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
 
 
# ============================================================================
#  CONFIG
# ============================================================================
 
DATA_DIR = Path(__file__).parent / "data"
REVIEWS_FILE  = DATA_DIR / "reviews_clean_no_exact_duplicates.csv"
PRODUCTS_FILE = DATA_DIR / "products_clean.csv"
OUTPUT_DIR    = Path("ncf_results")
 
RANDOM_SEED       = 42
TOP_N             = 10        # recommendations per user
MIN_REVIEWS_USER  = 3         # lowered: include more users (was 5)
LIKED_THRESHOLD   = 3.5       # lowered: more positives per user (was 4.0)
NEG_SAMPLE_RATIO  = 8         # doubled: richer negative signal (was 4)
 
# Model hyperparameters — tuned for small/sparse dataset (687 users, 2573 items)
EMBEDDING_DIM  = 16           # smaller: avoids overfitting (was 64)
MLP_LAYERS     = [64, 32]     # shallower: fewer params (was [128, 64, 32])
DROPOUT        = 0.3          # more regularisation (was 0.2)
LEARNING_RATE  = 5e-4         # slower: more stable on small data (was 1e-3)
WEIGHT_DECAY   = 1e-4         # stronger L2 penalty (was 1e-5)
EPOCHS         = 50           # more room to learn (was 20)
BATCH_SIZE     = 256          # smaller batches: more updates per epoch (was 1024)
EVAL_EVERY     = 1            # evaluate on val set every N epochs
PATIENCE       = 10           # longer patience before early stop (was 5)
 
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
# ============================================================================
#  STEP 1 — DATA LOADING & ENCODING
# ============================================================================
 
def load_and_encode(reviews_path: Path) -> tuple:
    print("\n[DATA]  Loading reviews ...")
    reviews = pd.read_csv(
        reviews_path,
        usecols=["user_id", "parent_asin", "rating", "timestamp_unix", "review_year"],
        low_memory=False,
    )
    reviews["rating"]         = pd.to_numeric(reviews["rating"],         errors="coerce")
    reviews["timestamp_unix"] = pd.to_numeric(reviews["timestamp_unix"], errors="coerce")
    reviews["review_year"]    = pd.to_numeric(reviews["review_year"],    errors="coerce")
    reviews = reviews.dropna(subset=["user_id", "parent_asin", "rating"])
    reviews["user_id"]     = reviews["user_id"].astype(str)
    reviews["parent_asin"] = reviews["parent_asin"].astype(str)
    reviews = reviews.sort_values("timestamp_unix").reset_index(drop=True)
 
    # Filter sparse users
    user_counts = reviews["user_id"].value_counts()
    active_users = user_counts[user_counts >= MIN_REVIEWS_USER].index
    reviews = reviews[reviews["user_id"].isin(active_users)].reset_index(drop=True)
 
    # Integer encode users and items
    users_list = reviews["user_id"].unique().tolist()
    items_list = reviews["parent_asin"].unique().tolist()
    user2idx = {u: i for i, u in enumerate(users_list)}
    item2idx = {it: i for i, it in enumerate(items_list)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}
 
    reviews["user_idx"] = reviews["user_id"].map(user2idx)
    reviews["item_idx"] = reviews["parent_asin"].map(item2idx)
 
    n_users = len(users_list)
    n_items = len(items_list)
 
    print(f"   Active users: {n_users:,}  |  unique items: {n_items:,}  |  interactions: {len(reviews):,}")
    return reviews, user2idx, item2idx, idx2user, idx2item, n_users, n_items
 
 
# ============================================================================
#  STEP 2 — TRAIN / VALIDATION / TEST SPLIT (temporal)
# ============================================================================
 
def temporal_split(reviews: pd.DataFrame):
    """
    Per-user temporal split:
      train  = first 70% of each user's interactions
      val    = next 15%
      test   = last 15%
    """
    train_rows, val_rows, test_rows = [], [], []
 
    for _, grp in reviews.groupby("user_idx"):
        grp = grp.sort_values("timestamp_unix")
        n = len(grp)
        t1 = max(1, int(n * 0.70))
        t2 = max(t1 + 1, int(n * 0.85))
        train_rows.append(grp.iloc[:t1])
        if t2 > t1:
            val_rows.append(grp.iloc[t1:t2])
        test_rows.append(grp.iloc[t2:])
 
    train = pd.concat(train_rows).reset_index(drop=True)
    val   = pd.concat(val_rows).reset_index(drop=True)  if val_rows  else train.iloc[:0]
    test  = pd.concat(test_rows).reset_index(drop=True) if test_rows else train.iloc[:0]
 
    print(f"\n[SPLIT] train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test
 
 
# ============================================================================
#  STEP 3 — NEGATIVE SAMPLING DATASET
# ============================================================================
 
class NCFDataset(Dataset):
    """
    For each positive interaction (user, item, label=1), sample NEG_SAMPLE_RATIO
    random items the user has NOT interacted with (label=0).
    """
    def __init__(self, interactions: pd.DataFrame, n_items: int,
                 user_item_set: set, neg_ratio: int = NEG_SAMPLE_RATIO):
        self.n_items      = n_items
        self.user_item_set = user_item_set
        self.neg_ratio    = neg_ratio
 
        # Build positive pairs
        positives = interactions[interactions["rating"] >= LIKED_THRESHOLD][
            ["user_idx", "item_idx"]
        ].values.tolist()
 
        self.samples = []
        for u, i in positives:
            self.samples.append((u, i, 1.0))
            for _ in range(neg_ratio):
                neg = random.randint(0, n_items - 1)
                while (u, neg) in user_item_set:
                    neg = random.randint(0, n_items - 1)
                self.samples.append((u, neg, 0.0))
 
        random.shuffle(self.samples)
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        u, i, label = self.samples[idx]
        return (
            torch.tensor(u,     dtype=torch.long),
            torch.tensor(i,     dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )
 
 
# ============================================================================
#  STEP 4 — NCF MODEL (GMF + MLP fusion)
# ============================================================================
 
class NCF(nn.Module):
    """
    Neural Collaborative Filtering:
      GMF branch: element-wise product of user & item embeddings
      MLP branch: concatenated embeddings through fully-connected layers
      Output    : sigmoid(linear(concat(gmf_out, mlp_out)))
 
    Reference: He et al. (2017) https://arxiv.org/abs/1708.05031
    """
    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = EMBEDDING_DIM,
                 mlp_layers: list = MLP_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
 
        # GMF embeddings
        self.gmf_user_embed = nn.Embedding(n_users, embed_dim)
        self.gmf_item_embed = nn.Embedding(n_items, embed_dim)
 
        # MLP embeddings (separate set — let each branch learn independently)
        self.mlp_user_embed = nn.Embedding(n_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(n_items, embed_dim)
 
        # MLP tower
        mlp_input_dim = embed_dim * 2
        layers = []
        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
 
        # Final prediction layer
        self.predict = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid  = nn.Sigmoid()
 
        self._init_weights()
 
    def _init_weights(self):
        for emb in [self.gmf_user_embed, self.gmf_item_embed,
                    self.mlp_user_embed, self.mlp_item_embed]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)
 
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF branch
        gmf_u = self.gmf_user_embed(user_ids)
        gmf_i = self.gmf_item_embed(item_ids)
        gmf_out = gmf_u * gmf_i           # element-wise product
 
        # MLP branch
        mlp_u = self.mlp_user_embed(user_ids)
        mlp_i = self.mlp_item_embed(item_ids)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=1))
 
        # Concat + predict
        concat  = torch.cat([gmf_out, mlp_out], dim=1)
        logit   = self.predict(concat).squeeze(-1)
        return self.sigmoid(logit)
 
 
# ============================================================================
#  STEP 5 — TRAINING LOOP
# ============================================================================
 
def train_model(
    model: NCF,
    train_loader: DataLoader,
    val_df: pd.DataFrame,
    n_items: int,
    user_item_set: set,
    train_items_by_user: dict = None,
) -> list[dict]:
    """
    Full training loop with:
      - Binary Cross Entropy loss
      - Adam optimiser with weight decay
      - Per-epoch validation Hit Rate @ TOP_N
      - Learning rate scheduler (ReduceLROnPlateau)
      - Early stopping (patience=5)
    Returns epoch log (list of dicts → written to CSV).
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.BCELoss()
 
    epoch_log = []
    best_hr    = -1.0
    no_improve = 0
 
    print(f"\n[TRAIN] Starting training on {DEVICE}  |  {EPOCHS} epochs  |  batch={BATCH_SIZE}")
    print(f"        Embeddings={EMBEDDING_DIM}  |  MLP={MLP_LAYERS}  |  lr={LEARNING_RATE}  |  patience={PATIENCE}")
    print("-" * 70)
 
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
 
        loop = tqdm(train_loader, desc=f"Epoch {epoch:>3}/{EPOCHS}", ncols=90, leave=False)
        for user_ids, item_ids, labels in loop:
            user_ids = user_ids.to(DEVICE)
            item_ids = item_ids.to(DEVICE)
            labels   = labels.to(DEVICE)
 
            optimizer.zero_grad()
            preds = model(user_ids, item_ids)
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
 
            total_loss += loss.item()
            n_batches  += 1
            loop.set_postfix(loss=f"{loss.item():.4f}")
 
        avg_loss = total_loss / max(n_batches, 1)
 
        # Validation every EVAL_EVERY epochs
        val_hr, val_prec, val_recall = 0.0, 0.0, 0.0
        if epoch % EVAL_EVERY == 0 and len(val_df) > 0:
            val_hr, val_prec, val_recall = evaluate_model(
                model, val_df, n_items, user_item_set, TOP_N, sample_users=500,
                train_items_by_user=train_items_by_user,
            )
 
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_loss)
 
        log_entry = {
            "epoch":      epoch,
            "train_loss": round(avg_loss, 6),
            "val_hit_rate":  round(val_hr,     4),
            "val_precision": round(val_prec,   4),
            "val_recall":    round(val_recall, 4),
            "learning_rate": current_lr,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
        }
        epoch_log.append(log_entry)
 
        print(
            f"  Epoch {epoch:>3}  loss={avg_loss:.4f}  "
            f"HR@{TOP_N}={val_hr:.4f}  P@{TOP_N}={val_prec:.4f}  "
            f"R@{TOP_N}={val_recall:.4f}  lr={current_lr:.2e}"
        )
 
        # Early stopping
        if val_hr > best_hr:
            best_hr    = val_hr
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs. Best HR@{TOP_N}={best_hr:.4f}")
                break
 
    return epoch_log
 
 
# ============================================================================
#  STEP 6 — EVALUATION
# ============================================================================
 
@torch.no_grad()
def evaluate_model(
    model: NCF,
    eval_df: pd.DataFrame,
    n_items: int,
    train_user_item_set: set,
    top_k: int = TOP_N,
    sample_users: int = None,
    train_items_by_user: dict = None,
) -> tuple[float, float, float]:
    """
    Hit Rate@K, Precision@K, Recall@K on the eval set.
    Uses prebuilt train_items_by_user dict for fast masking.
    """
    model.eval()
 
    # Build ground truth: user → set of liked items in eval
    gt = eval_df[eval_df["rating"] >= LIKED_THRESHOLD].groupby("user_idx")["item_idx"].apply(set).to_dict()
    user_ids_eval = list(gt.keys())
 
    if sample_users and len(user_ids_eval) > sample_users:
        user_ids_eval = random.sample(user_ids_eval, sample_users)
 
    # Build train mask dict if not provided
    if train_items_by_user is None:
        train_items_by_user = {}
        for (u, i) in train_user_item_set:
            train_items_by_user.setdefault(u, set()).add(i)
 
    all_items = torch.arange(n_items, device=DEVICE)
    hits, precisions, recalls = [], [], []
 
    for uid in user_ids_eval:
        true_items = gt.get(uid, set())
        if not true_items:
            continue
 
        user_tensor = torch.tensor([uid] * n_items, dtype=torch.long, device=DEVICE)
        scores = model(user_tensor, all_items).cpu().numpy()
 
        # Mask out training items (fast dict lookup)
        for item_idx in train_items_by_user.get(uid, set()):
            if item_idx < n_items:
                scores[item_idx] = -1.0
 
        top_k_items = set(np.argsort(scores)[::-1][:top_k])
        n_hits = len(top_k_items & true_items)
 
        hits.append(1 if n_hits > 0 else 0)
        precisions.append(n_hits / top_k)
        recalls.append(n_hits / len(true_items))
 
    hr     = np.mean(hits)       if hits       else 0.0
    prec   = np.mean(precisions) if precisions else 0.0
    recall = np.mean(recalls)    if recalls    else 0.0
    return hr, prec, recall
 
 
@torch.no_grad()
def generate_recommendations(
    model: NCF,
    user_idxs: list,
    n_items: int,
    train_user_item_set: set,
    idx2user: dict,
    idx2item: dict,
    top_k: int = TOP_N,
) -> pd.DataFrame:
    """Score all items for every user and return top-K recommendations."""
    model.eval()
    all_items  = torch.arange(n_items, device=DEVICE)
    rec_rows   = []
 
    print(f"\n[RECS]  Generating top-{top_k} recommendations for {len(user_idxs):,} users ...")
 
    for uid in tqdm(user_idxs, ncols=80, desc="  Scoring"):
        user_tensor = torch.tensor([uid] * n_items, dtype=torch.long, device=DEVICE)
        scores = model(user_tensor, all_items).cpu().numpy()
 
        # Mask items already interacted with
        train_items = {i for (u, i) in train_user_item_set if u == uid}
        for i in train_items:
            if i < n_items:
                scores[i] = -1.0
 
        top_k_idx = np.argsort(scores)[::-1][:top_k]
        for rank, item_idx in enumerate(top_k_idx, start=1):
            rec_rows.append({
                "user_id":        idx2user[uid],
                "rank":           rank,
                "parent_asin":    idx2item[item_idx],
                "predicted_score": round(float(scores[item_idx]), 6),
                "model":          "NCF",
            })
 
    return pd.DataFrame(rec_rows)
 
 
# ============================================================================
#  STEP 7 — POPULARITY BASELINE (for comparison)
# ============================================================================
 
def popularity_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        n_items: int, top_k: int = TOP_N) -> dict:
    """Simple popularity baseline: recommend globally most-purchased items."""
    pop_items = (
        train_df[train_df["rating"] >= LIKED_THRESHOLD]["item_idx"]
        .value_counts()
        .index.tolist()[:top_k]
    )
    gt = test_df[test_df["rating"] >= LIKED_THRESHOLD].groupby("user_idx")["item_idx"].apply(set).to_dict()
    train_seen = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
 
    hits, precs, recs = [], [], []
    for uid, true_items in gt.items():
        seen = train_seen.get(uid, set())
        pops = [i for i in pop_items if i not in seen][:top_k]
        n_hits = len(set(pops) & true_items)
        hits.append(1 if n_hits > 0 else 0)
        precs.append(n_hits / top_k)
        recs.append(n_hits / len(true_items) if true_items else 0)
 
    return {
        "model":         "Popularity Baseline",
        "hit_rate":      round(np.mean(hits),  4),
        "precision_at_k": round(np.mean(precs), 4),
        "recall_at_k":   round(np.mean(recs),  4),
        "users_evaluated": len(hits),
    }
 
 
# ============================================================================
#  MAIN
# ============================================================================
 
 
def _safe_save(df: pd.DataFrame, path: Path) -> None:
    """Write CSV via a temp file then rename — avoids PermissionError on Windows
    even if the target is open in Excel or another viewer."""
    import tempfile, shutil
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass
    try:
        shutil.move(str(tmp), str(path))
    except Exception:
        tmp.replace(path)
 
 
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  Neural Collaborative Filtering — Training Pipeline")
    print(f"  Device: {DEVICE}  |  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
 
    # 1. Load & encode
    reviews, user2idx, item2idx, idx2user, idx2item, n_users, n_items = load_and_encode(REVIEWS_FILE)
 
    # 2. Temporal split
    train_df, val_df, test_df = temporal_split(reviews)
 
    # 3. Build user-item set + fast per-user lookup dict
    train_user_item_set = set(
        zip(train_df["user_idx"].tolist(), train_df["item_idx"].tolist())
    )
    train_items_by_user: dict = {}
    for u, i in train_user_item_set:
        train_items_by_user.setdefault(u, set()).add(i)
    print(f"   Train mask dict built for {len(train_items_by_user):,} users.")
 
    # 4. Build NCF dataset + DataLoader
    print("\n[DATA]  Building NCF training dataset with negative sampling ...")
    ncf_dataset = NCFDataset(train_df, n_items, train_user_item_set, neg_ratio=NEG_SAMPLE_RATIO)
    train_loader = DataLoader(
        ncf_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    print(f"   Training samples: {len(ncf_dataset):,}  (positives + {NEG_SAMPLE_RATIO}x negatives)")
 
    # 5. Build model
    model = NCF(
        n_users=n_users,
        n_items=n_items,
        embed_dim=EMBEDDING_DIM,
        mlp_layers=MLP_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] NCF built  |  trainable params: {total_params:,}")
    print(model)
 
    # 6. Train
    epoch_log = train_model(
        model, train_loader, val_df, n_items,
        train_user_item_set, train_items_by_user,
    )
 
    # 7. Load best checkpoint
    best_ckpt = OUTPUT_DIR / "best_model.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        print(f"\n[CKPT]  Best model loaded from {best_ckpt}")
 
    # 8. Save training log CSV
    log_df = pd.DataFrame(epoch_log)
    log_path = OUTPUT_DIR / "ncf_training_log.csv"
    _safe_save(log_df, log_path)
    print(f"\n[SAVE]  Training log -> {log_path}")
 
    # 9. Final test evaluation (NCF)
    print("\n[EVAL]  Evaluating on test set ...")
    test_hr, test_prec, test_recall = evaluate_model(
        model, test_df, n_items, train_user_item_set,
        TOP_N, sample_users=None,
        train_items_by_user=train_items_by_user,
    )
    print(f"  NCF   HR@{TOP_N}={test_hr:.4f}  P@{TOP_N}={test_prec:.4f}  R@{TOP_N}={test_recall:.4f}")
 
    # 10. Popularity baseline evaluation
    pop_metrics = popularity_baseline(train_df, test_df, n_items, TOP_N)
    print(f"  Pop   HR@{TOP_N}={pop_metrics['hit_rate']:.4f}  "
          f"P@{TOP_N}={pop_metrics['precision_at_k']:.4f}  "
          f"R@{TOP_N}={pop_metrics['recall_at_k']:.4f}")
 
    # 11. Save per-user detailed evaluation
    print("\n[EVAL]  Building per-user evaluation detail ...")
    gt = test_df[test_df["rating"] >= LIKED_THRESHOLD].groupby("user_idx")["item_idx"].apply(set).to_dict()
    all_items_t = torch.arange(n_items, device=DEVICE)
    detail_rows = []
    model.eval()
    with torch.no_grad():
        for uid, true_items in tqdm(gt.items(), ncols=80, desc="  Per-user eval"):
            user_tensor = torch.tensor([uid] * n_items, dtype=torch.long, device=DEVICE)
            scores = model(user_tensor, all_items_t).cpu().numpy()
            for i in train_items_by_user.get(uid, set()):
                if i < n_items:
                    scores[i] = -1.0
            top_k_idx = set(np.argsort(scores)[::-1][:TOP_N])
            n_hits = len(top_k_idx & true_items)
            detail_rows.append({
                "user_id":         idx2user[uid],
                "hit":             int(n_hits > 0),
                "precision_at_k":  round(n_hits / TOP_N, 6),
                "recall_at_k":     round(n_hits / len(true_items), 6),
                "n_true_items":    len(true_items),
                "n_hits":          n_hits,
                "model":           "NCF",
            })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = OUTPUT_DIR / "ncf_evaluation_detail.csv"
    _safe_save(detail_df, detail_path)
    print(f"[SAVE]  Per-user detail -> {detail_path}")
 
    # 12. Generate & save user recommendations  ← DASHBOARD: rec viewer
    all_user_idxs = list(idx2user.keys())
    recs_df = generate_recommendations(
        model, all_user_idxs, n_items, train_user_item_set,
        idx2user, idx2item, top_k=TOP_N
    )
    recs_path = OUTPUT_DIR / "ncf_user_recommendations.csv"
    _safe_save(recs_df, recs_path)
    print(f"[SAVE]  User recommendations → {recs_path}  ({len(recs_df):,} rows)")
 
    # 13. Summary CSV  ← DASHBOARD: model comparison chart
    summary = pd.DataFrame([
        {
            "Model":             "NCF (Neural Collaborative Filtering)",
            "Architecture":      f"GMF+MLP  emb={EMBEDDING_DIM}  layers={MLP_LAYERS}",
            "Epochs_trained":    len(epoch_log),
            "Best_train_loss":   round(min(e["train_loss"] for e in epoch_log), 6),
            f"HitRate@{TOP_N}":  round(test_hr,     4),
            f"Precision@{TOP_N}":round(test_prec,   4),
            f"Recall@{TOP_N}":   round(test_recall, 4),
            "Users_evaluated":   len(detail_rows),
        },
        {
            "Model":             "Popularity Baseline",
            "Architecture":      "Frequency ranking",
            "Epochs_trained":    0,
            "Best_train_loss":   None,
            f"HitRate@{TOP_N}":  pop_metrics["hit_rate"],
            f"Precision@{TOP_N}":pop_metrics["precision_at_k"],
            f"Recall@{TOP_N}":   pop_metrics["recall_at_k"],
            "Users_evaluated":   pop_metrics["users_evaluated"],
        },
    ])
    summary_path = OUTPUT_DIR / "ncf_summary.csv"
    _safe_save(summary, summary_path)
    print(f"[SAVE]  Summary → {summary_path}")
 
    # 14. Print final report
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\n[DONE]  All outputs saved to:", OUTPUT_DIR.resolve())
    print("  → ncf_training_log.csv        (loss curve per epoch)")
    print("  → ncf_user_recommendations.csv (top-10 recs per user)")
    print("  → ncf_summary.csv             (model comparison)")
    print("  → ncf_evaluation_detail.csv   (per-user hit/precision/recall)")
 
 
if __name__ == "__main__":
    main()
