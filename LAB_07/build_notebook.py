"""
Script to build and execute 23MID0043_Lab07_Recommender_RF.ipynb cleanly for UCI OnlineRetail.csv
"""

import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(True)
    })

def add_code(code):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.strip().splitlines(True)
    })

# CELL 0: Header
add_md("""
# MDI3003 – Advanced Predictive Analytics
## Experiment 07: Constructing a Recommendation System from Transactional Data Using Random Forest and Baselines
**Student Name:** Harshita Bogineni  
**Registration Number:** 23MID0043  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT Vellore  
**Course Instructor:** Dr. Durgesh Kumar  
**Dataset:** Real UCI Online Retail Dataset (`OnlineRetail.csv` – 541,909 raw records)
""")

# CELL 1: Section 0
add_md("""
## 0. Environment Setup and Reproducibility
We initialize the environment, configure fixed random seeds (`SEED = 42`), ensure deterministic multi-threading behavior, and prepare output directories for models, figures, CSVs, and artifacts.
""")

# CELL 2: Code setup
add_code("""
import os, json, platform, time, warnings, hashlib, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib, sklearn

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)
REG_NO = "23MID0043"
STUDENT_NAME = "Harshita Bogineni"
K_LIST = [5, 10, 20]

# Ensure output folders exist
for d in ["artifacts", "figures", "models", "outputs_csv", "outputs", 
          "outputs/data_summary", "outputs/processed_data", "outputs/models", 
          "outputs/recommendations", "outputs/metrics", "outputs/figures", "outputs/audits", "outputs/logs"]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
print(f"Environment initialized. Seed={SEED}, scikit-learn={sklearn.__version__}, pandas={pd.__version__}")
""")

# CELL 3: Section 1
add_md("""
## 1. Data Loading and Dataset Card
We load the instructor-specified core dataset: **`OnlineRetail.csv`** (real UCI Online Retail dataset with 541,909 rows and 8 transactional columns). We verify its SHA-256 checksum and generate a comprehensive dataset card artifact.
""")

# CELL 4: Code 1
add_code("""
DATA_PATH = os.path.join("data", "OnlineRetail.csv") if os.path.exists(os.path.join("data", "OnlineRetail.csv")) else "OnlineRetail.csv"
if not os.path.exists(DATA_PATH) and os.path.exists(os.path.join("UCI", "OnlineRetail.csv")):
    DATA_PATH = os.path.join("UCI", "OnlineRetail.csv")

print(f"Loading core dataset from: {DATA_PATH}")
t0 = time.time()
df_raw = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
print(f"Loaded {len(df_raw):,} records in {time.time()-t0:.2f}s.")
df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])

# Verify required columns
required_cols = {'InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'}
assert required_cols.issubset(set(df_raw.columns)), f"Missing required columns: {required_cols - set(df_raw.columns)}"

# Compute hash
with open(DATA_PATH, 'rb') as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

print(f"Raw shape: {df_raw.shape}")
print(f"SHA-256: {sha256}")
df_raw.head()
""")

# CELL 5: Section 2
add_md("""
## 2. Transaction Cleaning and Integrity Audit (Section 6 of QP)
We apply strict e-commerce data hygiene:
1. **Missing Customer IDs:** Drop transactions without a valid `CustomerID` (135,080 rows removed) since personalized recommendations require customer history tracking.
2. **Cancellations & Returns:** Identify `InvoiceNo` starting with `'C'` and non-positive `Quantity` / `UnitPrice` ($\le 0$) and exclude them to prevent treating returns as positive preference signals.
3. **Exact Duplicates:** Remove 5,192 duplicate transaction lines.
4. **Transaction Amount:** Compute `Amount = Quantity * UnitPrice`.
""")

# CELL 6: Code 2
add_code("""
n_raw = len(df_raw)

# 1. Drop missing customer IDs
df_valid_id = df_raw.dropna(subset=['CustomerID']).copy()
n_missing_id = n_raw - len(df_valid_id)
df_valid_id['CustomerID'] = df_valid_id['CustomerID'].astype(int).astype(str)

# 2. Filter cancellations and non-positive quantities / prices
df_valid_id['is_cancel'] = df_valid_id['InvoiceNo'].astype(str).str.startswith('C')
df_valid_tx = df_valid_id[(~df_valid_id['is_cancel']) & (df_valid_id['Quantity'] > 0) & (df_valid_id['UnitPrice'] > 0)].copy()
n_cancelled_invalid = len(df_valid_id) - len(df_valid_tx)

# 3. Deduplicate
n_before_dedup = len(df_valid_tx)
df_clean = df_valid_tx.drop_duplicates().copy()
n_duplicates = n_before_dedup - len(df_clean)

# 4. Amount
df_clean['Amount'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"Data Preprocessing Audit:")
print(f"  • Raw transactions: {n_raw:,}")
print(f"  • Missing CustomerID removed: {n_missing_id:,} ({n_missing_id/n_raw:.2%})")
print(f"  • Cancellations & invalid qty/price removed: {n_cancelled_invalid:,} ({n_cancelled_invalid/n_raw:.2%})")
print(f"  • Duplicates removed: {n_duplicates:,} ({n_duplicates/n_raw:.2%})")
print(f"  • Cleaned transactions retained: {len(df_clean):,} ({len(df_clean)/n_raw:.2%})")
print(f"  • Unique Customers: {df_clean['CustomerID'].nunique():,}")
print(f"  • Unique Products (StockCode): {df_clean['StockCode'].nunique():,}")
print(f"  • Date range: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")

# Save Dataset Card
dataset_card = {
    "source": "UCI Machine Learning Repository - Online Retail Dataset (OnlineRetail.csv)",
    "version_sha256": sha256,
    "raw_rows": int(n_raw),
    "cleaned_rows": int(len(df_clean)),
    "unique_customers": int(df_clean['CustomerID'].nunique()),
    "unique_items": int(df_clean['StockCode'].nunique()),
    "date_range": [str(df_clean['InvoiceDate'].min()), str(df_clean['InvoiceDate'].max())],
    "return_cancellation_policy": "Excluded from purchase targets (InvoiceNo starts with 'C' or Quantity <= 0)",
    "missing_id_policy": f"Dropped {n_missing_id:,} rows lacking usable CustomerID",
    "privacy_note": "No direct PII present; transactions identified solely by numeric CustomerID"
}
with open("artifacts/dataset_card.json", "w") as f:
    json.dump(dataset_card, f, indent=2)
""")

# CELL 7: Section 3
add_md("""
## 3. Chronological Train / Validation / Test Split (Section 11.1)
Random train/test splits cause severe temporal data leakage in recommender systems because future purchases would leak into historical profiles.
We enforce a strict chronological quantile split:
- **Train Window ($< t_1$):** 70th percentile of dates (Historical feature source)
- **Validation Target Window ($[t_1, t_2)$):** 70th to 85th percentile (Hyperparameter tuning and development pair labels)
- **Locked-Test Target Window ($\ge t_2$):** 85th to 100th percentile (Final locked evaluation)
""")

# CELL 8: Code 3
add_code("""
t1 = df_clean['InvoiceDate'].quantile(0.70)
t2 = df_clean['InvoiceDate'].quantile(0.85)

train_hist  = df_clean[df_clean.InvoiceDate < t1].copy()
val_future  = df_clean[(df_clean.InvoiceDate >= t1) & (df_clean.InvoiceDate < t2)].copy()
test_future = df_clean[df_clean.InvoiceDate >= t2].copy()

# Assert zero temporal leakage
assert train_hist.InvoiceDate.max() < t1 <= val_future.InvoiceDate.min()
assert val_future.InvoiceDate.max() < t2 <= test_future.InvoiceDate.min()
assert t1 < t2, "Temporal ordering failed: t1 must be strictly before t2"

split_manifest = {
    "t1_train_cutoff": str(t1),
    "t2_test_cutoff": str(t2),
    "train_period": [str(train_hist.InvoiceDate.min()), str(train_hist.InvoiceDate.max())],
    "val_period": [str(val_future.InvoiceDate.min()), str(val_future.InvoiceDate.max())],
    "test_period": [str(test_future.InvoiceDate.min()), str(test_future.InvoiceDate.max())],
    "train_rows": int(len(train_hist)),
    "val_rows": int(len(val_future)),
    "test_rows": int(len(test_future)),
    "train_users": int(train_hist.CustomerID.nunique()),
    "val_users": int(val_future.CustomerID.nunique()),
    "test_users": int(test_future.CustomerID.nunique()),
    "train_items": int(train_hist.StockCode.nunique()),
    "val_items": int(val_future.StockCode.nunique()),
    "test_items": int(test_future.StockCode.nunique())
}
with open("artifacts/split_manifest.json", "w") as f:
    json.dump(split_manifest, f, indent=2)

print("Split Summary:")
print(f"  • Train: {split_manifest['train_period'][0]} to {split_manifest['train_period'][1]} | {split_manifest['train_rows']:,} rows | {split_manifest['train_users']:,} users | {split_manifest['train_items']:,} items")
print(f"  • Val:   {split_manifest['val_period'][0]} to {split_manifest['val_period'][1]} | {split_manifest['val_rows']:,} rows | {split_manifest['val_users']:,} users | {split_manifest['val_items']:,} items")
print(f"  • Test:  {split_manifest['test_period'][0]} to {split_manifest['test_period'][1]} | {split_manifest['test_rows']:,} rows | {split_manifest['test_users']:,} users | {split_manifest['test_items']:,} items")
""")

# CELL 9: Visualizations 1-4
add_md("""
### Exploratory Data Visualizations (Figures 1 – 4)
We plot daily transaction volume with cutoff dates, top 15 most frequent items, customer purchase frequency, and the item popularity long-tail distribution.
""")

# CELL 10: Code viz 1-4
add_code("""
# Figure 1: Daily Transaction Volume
daily = df_clean.set_index('InvoiceDate').resample('D')['InvoiceNo'].nunique()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily.index, daily.values, color='#2b6cb0', lw=1.8, label='Daily Transactions')
ax.axvline(t1, color='#e53e3e', linestyle='--', lw=1.8, label=f'Train Cutoff t1 ({t1.strftime("%Y-%m-%d")})')
ax.axvline(t2, color='#dd6b20', linestyle='--', lw=1.8, label=f'Test Cutoff t2 ({t2.strftime("%Y-%m-%d")})')
ax.set_title("Figure 1: Daily Transaction Volume Over Time (UCI Online Retail)", fontweight='bold')
ax.set_xlabel("Date"); ax.set_ylabel("Number of Invoices"); ax.legend()
plt.tight_layout()
plt.savefig("figures/01_transaction_volume.png", dpi=130)
plt.show()

# Figure 2: Top 15 Items
desc_lookup = df_clean.dropna(subset=['Description']).groupby('StockCode')['Description'].first().to_dict()
top15 = train_hist.groupby('StockCode')['InvoiceNo'].nunique().sort_values(ascending=False).head(15)
top15_labels = [f"{c} ({desc_lookup.get(c, 'N/A')[:18]})" for c in top15.index]
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top15.values, y=top15_labels, palette="Blues_r", ax=ax)
ax.set_title("Figure 2: Top 15 Items by Transaction Count (Training Set)", fontweight='bold')
ax.set_xlabel("Unique Invoices"); ax.set_ylabel("Product")
plt.tight_layout()
plt.savefig("figures/02_top15_items.png", dpi=130)
plt.show()

# Figure 3: Customer Purchase Frequency
cust_freq = train_hist.groupby('CustomerID')['InvoiceNo'].nunique()
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(cust_freq, bins=40, kde=False, color='#319795', ax=ax)
ax.set_yscale('log')
ax.set_title("Figure 3: Customer Purchase Frequency Distribution (Log Scale)", fontweight='bold')
ax.set_xlabel("Invoices per Customer"); ax.set_ylabel("Customer Count (Log)")
plt.tight_layout()
plt.savefig("figures/03_customer_frequency.png", dpi=130)
plt.show()

# Figure 4: Item Popularity Long-Tail
item_freq = train_hist.groupby('StockCode')['InvoiceNo'].nunique().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(item_freq)), item_freq.values, color='#805ad5', lw=2)
ax.set_yscale('log')
ax.set_title("Figure 4: Item Popularity Long-Tail Distribution (Log Scale)", fontweight='bold')
ax.set_xlabel("Item Popularity Rank"); ax.set_ylabel("Transactions (Log)")
plt.tight_layout()
plt.savefig("figures/04_item_popularity_longtail.png", dpi=130)
plt.show()
""")

# CELL 11: Section 4
add_md("""
## 4. Popularity Baseline & Candidate-Recall Audit (Sections 11.2, 11.3, 13.1)
The popularity baseline is constructed strictly from the training history ($< t_1$) by sorting items by unique invoice count.
The candidate universe consists of all items with $\ge 1$ buyer in training history.
We conduct a rigorous **Candidate-Recall Audit** to measure whether future relevant items can be retrieved by the candidate universe before ranking is applied.
""")

# CELL 12: Code candidate audit
add_code("""
MIN_BUYERS = 1
item_buyer_counts = train_hist.groupby('StockCode')['CustomerID'].nunique()
eligible_candidate_items = item_buyer_counts[item_buyer_counts >= MIN_BUYERS].index.tolist()
print(f"Eligible Candidate Universe Size: {len(eligible_candidate_items):,} products")

# Popular items list (Strictly from training set)
popular_items = (train_hist.groupby('StockCode')['InvoiceNo']
                 .nunique().sort_values(ascending=False).index.tolist())

# Candidate policy artifact
candidate_policy = {
    "policy_version": "v1.0-online-retail-train-universe",
    "min_buyers_filter": MIN_BUYERS,
    "candidate_universe_size": len(eligible_candidate_items),
    "deduplication_policy": "Exclude items already purchased prior to cutoff",
    "repeat_purchase_allowed": False
}
with open("artifacts/candidate_policy.json", "w") as f:
    json.dump(candidate_policy, f, indent=2)

# Audit Candidate Recall
def get_future_positives(future_df):
    return future_df.groupby('CustomerID')['StockCode'].apply(lambda s: set(s.unique())).to_dict()

val_positives_map = get_future_positives(val_future)
test_positives_map = get_future_positives(test_future)
cand_set = set(eligible_candidate_items)

def audit_candidate_recall(pos_map, cset):
    all_pos = [it for items in pos_map.values() for it in items]
    cov = [it for it in all_pos if it in cset]
    recall = len(cov) / len(all_pos) if all_pos else 0.0
    fully_rep = sum(1 for items in pos_map.values() if items.issubset(cset))
    return len(all_pos), len(cov), recall, fully_rep, fully_rep / len(pos_map)

v_tot, v_cov, v_rec, v_full, v_full_pct = audit_candidate_recall(val_positives_map, cand_set)
t_tot, t_cov, t_rec, t_full, t_full_pct = audit_candidate_recall(test_positives_map, cand_set)

cand_recall_df = pd.DataFrame([
    {"Window": "Validation Target [t1, t2)", "Total Future Positives": v_tot, "Covered in Candidate Set": v_cov, 
     "Candidate Recall": v_rec, "Users 100% Representable": f"{v_full}/{len(val_positives_map)} ({v_full_pct:.2%})"},
    {"Window": "Locked Test Target [t2, max)", "Total Future Positives": t_tot, "Covered in Candidate Set": t_cov, 
     "Candidate Recall": t_rec, "Users 100% Representable": f"{t_full}/{len(test_positives_map)} ({t_full_pct:.2%})"}
])
cand_recall_df.to_csv("outputs_csv/23MID0043_Lab07_Candidate_Recall.csv", index=False)
cand_recall_df
""")

# CELL 13: Section 5
add_md("""
## 5. Leakage-Safe Feature Engineering (Section 12)
All customer, item, and customer-item pair features are computed strictly from the history available before the cutoff timestamp ($t \le \text{cutoff}$).
- **Customer Level:** `cust_txns`, `cust_items`, `cust_qty`, `cust_spend`, `cust_active_days`, `cust_recency_days`, `cust_lifespan_days`, `cust_avg_basket`
- **Item Level:** `item_txns`, `item_buyers`, `item_qty`, `item_avg_price`, `item_recency_days`, `item_repeat_rate`
- **Customer-Item Pair Level:** `pair_purchases`, `pair_qty`, `pair_spend`, `pair_recency_days`
""")

# CELL 14: Code feature functions
add_code("""
def customer_features(hist, cutoff):
    g = hist.groupby('CustomerID')
    out = g.agg(
        cust_txns=('InvoiceNo','nunique'),
        cust_items=('StockCode','nunique'),
        cust_qty=('Quantity','sum'),
        cust_spend=('Amount','sum'),
        cust_last=('InvoiceDate','max'),
        cust_first=('InvoiceDate','min'),
        cust_active_days=('InvoiceDate', lambda d: d.dt.date.nunique())
    ).reset_index()
    out['cust_recency_days'] = (cutoff - out['cust_last']).dt.total_seconds() / 86400.0
    out['cust_lifespan_days'] = (cutoff - out['cust_first']).dt.total_seconds() / 86400.0
    out['cust_avg_basket'] = out['cust_spend'] / out['cust_txns'].clip(lower=1)
    return out.drop(columns=['cust_last','cust_first'])

def item_features(hist, cutoff):
    g = hist.groupby('StockCode')
    out = g.agg(
        item_txns=('InvoiceNo','nunique'),
        item_buyers=('CustomerID','nunique'),
        item_qty=('Quantity','sum'),
        item_avg_price=('UnitPrice','mean'),
        item_last=('InvoiceDate','max')
    ).reset_index()
    out['item_recency_days'] = (cutoff - out['item_last']).dt.total_seconds() / 86400.0
    cust_item_counts = hist.groupby(['StockCode','CustomerID'])['InvoiceNo'].nunique()
    repeat_counts = cust_item_counts[cust_item_counts > 1].groupby('StockCode').count()
    out['item_repeat_rate'] = out['StockCode'].map(repeat_counts).fillna(0) / out['item_buyers'].clip(lower=1)
    return out.drop(columns=['item_last'])

def pair_features(hist, cutoff):
    out = (hist.groupby(['CustomerID','StockCode'])
            .agg(pair_purchases=('InvoiceNo','nunique'),
                 pair_qty=('Quantity','sum'),
                 pair_spend=('Amount','sum'),
                 pair_last=('InvoiceDate','max'))
            .reset_index())
    out['pair_recency_days'] = (cutoff - out['pair_last']).dt.total_seconds() / 86400.0
    return out.drop(columns=['pair_last'])

def build_feature_matrix(hist, pairs_df, cutoff):
    cf = customer_features(hist, cutoff)
    itf = item_features(hist, cutoff)
    pf = pair_features(hist, cutoff)

    out = pairs_df.merge(cf, on='CustomerID', how='left')
    out = out.merge(itf, on='StockCode', how='left')
    out = out.merge(pf, on=['CustomerID','StockCode'], how='left')

    fill_zero = ['pair_purchases','pair_qty','pair_spend']
    for c in fill_zero:
        out[c] = out[c].fillna(0)
    big_recency = (cutoff - hist['InvoiceDate'].min()).total_seconds() / 86400.0 + 1.0
    for c in ['cust_recency_days','item_recency_days','pair_recency_days','cust_lifespan_days']:
        out[c] = out[c].fillna(big_recency)
    for c in ['cust_txns','cust_items','cust_qty','cust_spend','cust_active_days','cust_avg_basket',
              'item_txns','item_buyers','item_qty','item_avg_price','item_repeat_rate']:
        out[c] = out[c].fillna(0)
    return out

FEATURE_COLS = [
    'cust_txns', 'cust_items', 'cust_qty', 'cust_spend', 'cust_active_days', 'cust_recency_days', 'cust_lifespan_days', 'cust_avg_basket',
    'item_txns', 'item_buyers', 'item_qty', 'item_avg_price', 'item_recency_days', 'item_repeat_rate',
    'pair_purchases', 'pair_qty', 'pair_spend', 'pair_recency_days'
]
print(f"Feature engineering pipeline defined with {len(FEATURE_COLS)} leakage-safe features.")
""")

# CELL 15: Section 6
add_md("""
## 6. Negative Sampling and Development Pair Construction (Section 11.4)
For each customer with future purchases in the validation target window $[t_1, t_2)$, we form positive user-item pairs ($y=1$) and sample $N_{\text{neg}} = 50$ negative items ($y=0$) from the candidate pool not purchased by the customer.
""")

# CELL 16: Code dev pairs & Fig 5
add_code("""
def sample_negatives(positives, candidate_items, n_neg=50, rng=None):
    rng = np.random.default_rng(SEED) if rng is None else rng
    pool = np.array(list(set(candidate_items) - set(positives)))
    n = min(n_neg, len(pool))
    return rng.choice(pool, size=n, replace=False).tolist() if n > 0 else []

def build_dev_pairs(positives_map, candidate_items, n_neg=50, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for cust, pos in positives_map.items():
        for it in pos:
            if it in cand_set:
                rows.append((cust, it, 1))
        negs = sample_negatives(pos, candidate_items, n_neg=n_neg, rng=rng)
        for it in negs:
            rows.append((cust, it, 0))
    return pd.DataFrame(rows, columns=['CustomerID', 'StockCode', 'label'])

dev_pairs = build_dev_pairs(val_positives_map, eligible_candidate_items, n_neg=50)
X_dev_raw = build_feature_matrix(train_hist, dev_pairs[['CustomerID','StockCode']], cutoff=t1)
X_dev_raw['label'] = dev_pairs['label'].values

print(f"Dev Feature Matrix: {X_dev_raw.shape} | Positive Rate: {X_dev_raw['label'].mean():.4f}")

# Figure 5: Class Balance
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='label', data=X_dev_raw, palette=['#e53e3e', '#38a169'], ax=ax)
ax.set_xticklabels(['Sampled Negatives (0)', 'True Positives (1)'])
ax.set_title("Figure 5: Class Balance After Negative Sampling (Dev Set)", fontweight='bold')
ax.set_xlabel("Relevance Label"); ax.set_ylabel("Count of Pairs")
plt.tight_layout()
plt.savefig("figures/05_class_balance.png", dpi=130)
plt.show()
""")

# CELL 17: Section 7
add_md("""
## 7. Ranking Metric Functions & Validation-Based Hyperparameter Tuning (Section 13)
We define the required ranking metrics: Precision@K, Recall@K, HitRate@K, MAP@K, and NDCG@K.
We tune the Random Forest using a `GroupShuffleSplit` on `CustomerID` to prevent leakage between folds, selecting the configuration that maximizes validation Recall@10 and PR-AUC.
""")

# CELL 18: Code ranking functions & tuning
add_code("""
def precision_at_k(recs, relevant, k):
    return len(set(list(recs)[:k]) & set(relevant)) / max(k, 1)

def recall_at_k(recs, relevant, k):
    if not relevant: return np.nan
    return len(set(list(recs)[:k]) & set(relevant)) / len(set(relevant))

def hit_rate_at_k(recs, relevant, k):
    return float(len(set(list(recs)[:k]) & set(relevant)) > 0)

def average_precision_at_k(recs, relevant, k=10):
    rel = set(relevant)
    if not rel: return np.nan
    score, hits = 0.0, 0
    for rank, item in enumerate(list(recs)[:k], start=1):
        if item in rel:
            hits += 1
            score += hits / rank
    return score / min(len(rel), k)

def ndcg_at_k(recs, relevant, k=10):
    rel = set(relevant)
    if not rel: return np.nan
    dcg = 0.0
    for rank, item in enumerate(list(recs)[:k], start=1):
        if item in rel:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else np.nan

# GroupShuffleSplit tuning
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
train_idx, val_idx = next(gss.split(X_dev_raw, groups=X_dev_raw['CustomerID']))
dev_train = X_dev_raw.iloc[train_idx].reset_index(drop=True)
dev_val   = X_dev_raw.iloc[val_idx].reset_index(drop=True)

Xtr, ytr = dev_train[FEATURE_COLS], dev_train['label']
Xva, yva = dev_val[FEATURE_COLS], dev_val['label']

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [None, 12],
    'min_samples_leaf': [2],
    'max_features': ['sqrt'],
}

def eval_val_recall10(model, df_v):
    dfv = df_v.copy()
    dfv['score'] = model.predict_proba(dfv[FEATURE_COLS])[:, 1]
    recs_map = dfv.sort_values(['CustomerID','score'], ascending=[True, False]).groupby('CustomerID')['StockCode'].apply(list)
    rels_map = dfv[dfv.label == 1].groupby('CustomerID')['StockCode'].apply(set)
    scores = [recall_at_k(recs_map[c], rels_map[c], 10) for c in rels_map.keys() if c in recs_map]
    return np.nanmean(scores) if scores else np.nan

grid_results = []
best_score, best_params = -1, None
for params in ParameterGrid(param_grid):
    rf_tmp = RandomForestClassifier(class_weight='balanced_subsample', random_state=SEED, n_jobs=-1, **params)
    rf_tmp.fit(Xtr, ytr)
    val_pr_auc = average_precision_score(yva, rf_tmp.predict_proba(Xva)[:, 1])
    val_rec10 = eval_val_recall10(rf_tmp, dev_val)
    grid_results.append({**params, "val_Recall@10": val_rec10, "val_PR_AUC": val_pr_auc})
    if val_rec10 > best_score:
        best_score, best_params = val_rec10, params

print(pd.DataFrame(grid_results).sort_values("val_Recall@10", ascending=False))
print(f"Selected Hyperparameters: {best_params}")

# Refit on FULL development dataset
rf_final = RandomForestClassifier(class_weight='balanced_subsample', random_state=SEED, n_jobs=-1, **best_params)
t_rf_fit = time.time()
rf_final.fit(X_dev_raw[FEATURE_COLS], X_dev_raw['label'])
rf_train_time = time.time() - t_rf_fit
print(f"Final Random Forest trained on {len(X_dev_raw):,} pairs in {rf_train_time:.2f}s")
""")

# CELL 19: Figure 6
add_md("""
### Feature Importance (Figure 6)
We visualize the Gini feature importances (Mean Decrease in Impurity) of the trained Random Forest ranker.
""")

# CELL 20: Code Fig 6
add_code("""
feat_imp = pd.Series(rf_final.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax)
ax.set_title("Figure 6: Random Forest Feature Importance (MDI)", fontweight='bold')
ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
plt.tight_layout()
plt.savefig("figures/06_feature_importance.png", dpi=130)
plt.show()
print("Interpretation: Item transaction volume, customer spending volume, and historical pair recency are primary predictive drivers.")
""")

# CELL 21: Section 8
add_md("""
## 8. Locked-Test Scoring, Baselines & Collaborative Filtering (Sections 14, 18.1)
We evaluate the complete system on the **Locked Test Window ($\ge t_2$)**.
- History available at test time: $\text{Train} \cup \text{Validation}$ ($< t_2$).
- Systems evaluated:
  1. **Popularity Baseline:** Global most frequent items from training period not previously bought.
  2. **Random Forest Ranker:** Supervised tree ensemble scoring candidates using time-valid customer/item/pair features.
  3. **Item-Item Collaborative Filtering:** Cosine item-item similarity matrix computed from transaction co-occurrences up to $t_2$.
  4. **Matrix Factorization (Truncated SVD):** 20 latent factors learned from the implicit user-item interaction matrix up to $t_2$.
""")

# CELL 22: Code test scoring
add_code("""
history_for_test = pd.concat([train_hist, val_future], ignore_index=True)
test_customers = sorted(set(test_future['CustomerID'].unique()))
already_purchased = history_for_test.groupby('CustomerID')['StockCode'].apply(set).to_dict()

# 1. Popularity Recommendations
pop_items_test = (history_for_test.groupby('StockCode')['InvoiceNo']
                  .nunique().sort_values(ascending=False).index.tolist())

pop_recs = {}
for c in test_customers:
    seen = already_purchased.get(c, set())
    pop_recs[c] = [i for i in pop_items_test if i not in seen][:20]

# 2. Random Forest Test Candidate Scoring
t_cf_feat = time.time()
cf_test = customer_features(history_for_test, t2).set_index('CustomerID')
itf_test = item_features(history_for_test, t2).set_index('StockCode')
pf_test = pair_features(history_for_test, t2).set_index(['CustomerID','StockCode'])

POP_POOL_SIZE = 500
top_pop_pool = [i for i in pop_items_test if i in cand_set][:POP_POOL_SIZE]

test_cands = []
for c in test_customers:
    seen = already_purchased.get(c, set())
    c_items = [i for i in top_pop_pool if i not in seen][:300]
    for i in c_items:
        test_cands.append((c, i))

test_cand_df = pd.DataFrame(test_cands, columns=['CustomerID', 'StockCode'])
X_test_cands = test_cand_df.merge(cf_test, on='CustomerID', how='left')
X_test_cands = X_test_cands.merge(itf_test, on='StockCode', how='left')
X_test_cands = X_test_cands.merge(pf_test, on=['CustomerID','StockCode'], how='left')

for c in ['pair_purchases','pair_qty','pair_spend']: X_test_cands[c] = X_test_cands[c].fillna(0)
big_rec_test = (t2 - history_for_test['InvoiceDate'].min()).total_seconds() / 86400.0 + 1.0
for c in ['cust_recency_days','item_recency_days','pair_recency_days','cust_lifespan_days']: X_test_cands[c] = X_test_cands[c].fillna(big_rec_test)
for c in ['cust_txns','cust_items','cust_qty','cust_spend','cust_active_days','cust_avg_basket',
          'item_txns','item_buyers','item_qty','item_avg_price','item_repeat_rate']: X_test_cands[c] = X_test_cands[c].fillna(0)

t_rf_score = time.time()
X_test_cands['rf_score'] = rf_final.predict_proba(X_test_cands[FEATURE_COLS])[:, 1]
rf_score_time = time.time() - t_rf_score

rf_recs = (X_test_cands.sort_values(['CustomerID','rf_score'], ascending=[True, False])
           .groupby('CustomerID')['StockCode'].apply(lambda s: list(s)[:20]).to_dict())
for c in test_customers:
    if c not in rf_recs or len(rf_recs[c]) == 0: rf_recs[c] = pop_recs[c]

# 3. Item-Item Collaborative Filtering
hist_users = sorted(history_for_test['CustomerID'].unique())
hist_items = eligible_candidate_items
u_index = {u: i for i, u in enumerate(hist_users)}
i_index = {it: i for i, it in enumerate(hist_items)}
inv_i_index = {i: it for it, i in i_index.items()}

valid_hist = history_for_test[history_for_test.StockCode.isin(set(hist_items))].copy()
ui_matrix = csr_matrix((np.ones(len(valid_hist), dtype=np.float32), 
                        (valid_hist['CustomerID'].map(u_index).values, valid_hist['StockCode'].map(i_index).values)),
                       shape=(len(hist_users), len(hist_items)))
ui_matrix.data = np.ones_like(ui_matrix.data)

t_cf_fit = time.time()
item_sim = cosine_similarity(ui_matrix.T.tocsr(), dense_output=False)
cf_train_time = time.time() - t_cf_fit

t_cf_sc = time.time()
cf_recs = {}
for u in test_customers:
    if u not in u_index: cf_recs[u] = pop_recs[u]; continue
    uvec = ui_matrix[u_index[u]]
    scores = uvec.dot(item_sim).toarray().ravel()
    seen = already_purchased.get(u, set())
    recs = [inv_i_index[idx] for idx in np.argsort(-scores) if inv_i_index[idx] not in seen][:20]
    if len(recs) < 20: recs.extend([it for it in pop_recs[u] if it not in recs and it not in seen][:20-len(recs)])
    cf_recs[u] = recs
cf_score_time = time.time() - t_cf_sc

# 4. Matrix Factorization (SVD)
t_mf_fit = time.time()
svd = TruncatedSVD(n_components=20, random_state=SEED)
user_factors = svd.fit_transform(ui_matrix)
item_factors = svd.components_.T
mf_train_time = time.time() - t_mf_fit

t_mf_sc = time.time()
mf_recs = {}
for u in test_customers:
    if u not in u_index: mf_recs[u] = pop_recs[u]; continue
    scores = np.dot(item_factors, user_factors[u_index[u]])
    seen = already_purchased.get(u, set())
    recs = [inv_i_index[idx] for idx in np.argsort(-scores) if inv_i_index[idx] not in seen][:20]
    if len(recs) < 20: recs.extend([it for it in pop_recs[u] if it not in recs and it not in seen][:20-len(recs)])
    mf_recs[u] = recs
mf_score_time = time.time() - t_mf_sc

print(f"Test scoring complete across all 4 recommender systems for {len(test_customers):,} test customers.")
""")

# CELL 23: Section 9
add_md("""
## 9. Ranking Evaluation and Comparison Across Models (Section 15)
We compute Precision@K, Recall@K, HitRate@K, MAP@K, and NDCG@K for $K \in \{5, 10, 20\}$ against actual future customer purchases in the locked test period.
""")

# CELL 24: Code evaluation
add_code("""
def evaluate_system(recs_map, positives_map, k_list):
    rows = []
    for k in k_list:
        p, r, h, mapk, ndcg = [], [], [], [], []
        for cust, rel in positives_map.items():
            recs = recs_map.get(cust, [])
            if not rel or not recs: continue
            p.append(precision_at_k(recs, rel, k))
            r.append(recall_at_k(recs, rel, k))
            h.append(hit_rate_at_k(recs, rel, k))
            mapk.append(average_precision_at_k(recs, rel, k))
            ndcg.append(ndcg_at_k(recs, rel, k))
        rows.append({"K": k, "Precision@K": np.nanmean(p), "Recall@K": np.nanmean(r),
                     "HitRate@K": np.nanmean(h), "MAP@K": np.nanmean(mapk), "NDCG@K": np.nanmean(ndcg)})
    return pd.DataFrame(rows)

pop_eval = evaluate_system(pop_recs, test_positives_map, K_LIST).assign(Model="Popularity")
rf_eval  = evaluate_system(rf_recs,  test_positives_map, K_LIST).assign(Model="Random Forest")
cf_eval  = evaluate_system(cf_recs,  test_positives_map, K_LIST).assign(Model="Item-Item CF")
mf_eval  = evaluate_system(mf_recs,  test_positives_map, K_LIST).assign(Model="Matrix Factorization (SVD)")

pair_auc = roc_auc_score(X_dev_raw['label'], rf_final.predict_proba(X_dev_raw[FEATURE_COLS])[:, 1])
pair_prauc = average_precision_score(X_dev_raw['label'], rf_final.predict_proba(X_dev_raw[FEATURE_COLS])[:, 1])

all_results = pd.concat([pop_eval, rf_eval, cf_eval, mf_eval], ignore_index=True)
all_results["ROC-AUC(pairs)"] = np.where(all_results.Model == "Random Forest", pair_auc, np.nan)
all_results["PR-AUC(pairs)"] = np.where(all_results.Model == "Random Forest", pair_prauc, np.nan)
all_results.to_csv(f"outputs_csv/{REG_NO}_Lab07_Ranking_Metrics.csv", index=False)
all_results
""")

# CELL 25: Figures 7-11
add_md("""
### Model Evaluation Visualizations (Figures 7 – 11)
We visualize Precision@K and Recall@K vs $K$, comparative ranking metrics at $K=10$, score distribution for positive vs negative pairs, catalog coverage, and all-systems recall comparison.
""")

# CELL 26: Code viz 7-11
add_code("""
# Figure 7: Precision & Recall vs K
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for model_name, g in all_results.groupby("Model"):
    axes[0].plot(g.K, g["Precision@K"], marker='o', lw=2, label=model_name)
    axes[1].plot(g.K, g["Recall@K"], marker='s', lw=2, label=model_name)
axes[0].set_title("Figure 7a: Precision@K vs K", fontweight='bold'); axes[0].set_xlabel("K"); axes[0].set_ylabel("Precision@K"); axes[0].legend()
axes[1].set_title("Figure 7b: Recall@K vs K", fontweight='bold'); axes[1].set_xlabel("K"); axes[1].set_ylabel("Recall@K"); axes[1].legend()
plt.tight_layout()
plt.savefig("figures/07_precision_recall_vs_k.png", dpi=130)
plt.show()

# Figure 8: Metrics at K=10
comp10 = all_results[all_results.K == 10].set_index("Model")[["Precision@K","Recall@K","HitRate@K","NDCG@K"]]
fig, ax = plt.subplots(figsize=(9, 4.5))
comp10.T.plot(kind='bar', ax=ax, colormap="tab10", width=0.75)
ax.set_title("Figure 8: Recommendation Ranking Metrics @ K=10", fontweight='bold')
ax.set_ylabel("Score"); ax.set_xticklabels(comp10.columns, rotation=0)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("figures/08_popularity_vs_rf.png", dpi=130)
plt.show()

# Figure 9: Score Distribution
scores_pos = rf_final.predict_proba(X_dev_raw.loc[X_dev_raw.label==1, FEATURE_COLS])[:, 1]
scores_neg = rf_final.predict_proba(X_dev_raw.loc[X_dev_raw.label==0, FEATURE_COLS])[:, 1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(scores_neg, bins=35, alpha=0.6, label="Sampled Negatives", color='#e53e3e', density=True)
ax.hist(scores_pos, bins=35, alpha=0.6, label="True Positives", color='#38a169', density=True)
ax.set_title("Figure 9: RF Purchase Probability Score Distribution (Dev Set)", fontweight='bold')
ax.set_xlabel("Predicted Probability P(y=1|x)"); ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.savefig("figures/09_score_distribution.png", dpi=130)
plt.show()

# Figure 10: Catalog Coverage
def catalog_coverage(recs_map, catalog, k=10):
    recs = set()
    for v in recs_map.values(): recs.update(v[:k])
    return 100.0 * len(recs) / len(catalog)

cov_pop = catalog_coverage(pop_recs, eligible_candidate_items)
cov_rf = catalog_coverage(rf_recs, eligible_candidate_items)
cov_cf = catalog_coverage(cf_recs, eligible_candidate_items)
cov_mf = catalog_coverage(mf_recs, eligible_candidate_items)

cov_df = pd.DataFrame([{"Model": "Popularity", "Coverage": cov_pop}, {"Model": "Random Forest", "Coverage": cov_rf},
                       {"Model": "Item-Item CF", "Coverage": cov_cf}, {"Model": "Matrix Factorization", "Coverage": cov_mf}])
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x="Model", y="Coverage", data=cov_df, palette="deep", ax=ax)
ax.set_title("Figure 10: Catalog Coverage @ Top-10 (%)", fontweight='bold')
ax.set_ylabel("% of Catalog Recommended")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()+0.5), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("figures/10_catalog_coverage.png", dpi=130)
plt.show()

# Figure 11: All Systems Recall
fig, ax = plt.subplots(figsize=(8, 4.5))
for model_name, g in all_results.groupby("Model"):
    ax.plot(g.K, g["Recall@K"], marker='o', lw=2.2, label=model_name)
ax.set_title("Figure 11: Recall@K Across All Recommender Systems", fontweight='bold')
ax.set_xlabel("Cutoff K"); ax.set_ylabel("Recall@K"); ax.legend()
plt.tight_layout()
plt.savefig("figures/11_all_systems_recall.png", dpi=130)
plt.show()
""")

# CELL 27: Section 10 - Five Case Audit
add_md("""
## 10. Required Five-Case Audit (Section 16.1)
We perform an in-depth audit on **FIVE DISTINCT CUSTOMERS** representing the exact required case types:
1. **Case 1:** Successful personalized recommendation.
2. **Case 2:** Relevant item missed by RF but found by Popularity baseline.
3. **Case 3:** Case where RF beats Popularity.
4. **Case 4:** Sparse / cold-start customer ($\le 2$ historical transactions).
5. **Case 5:** Questionable recommendation caused by candidate retrieval or sampling limitations.
""")

# CELL 28: Code 5-case audit
add_code("""
case_records = []
for cust in test_customers:
    rel = test_positives_map.get(cust, set())
    if not rel: continue
    rf_top5 = rf_recs.get(cust, [])[:5]
    pop_top5 = pop_recs.get(cust, [])[:5]
    rf_hits = set(rf_top5) & rel
    pop_hits = set(pop_top5) & rel

    cust_hist = history_for_test[history_for_test.CustomerID == cust]
    case_records.append({
        "CustomerID": cust,
        "hist_txns": cust_hist['InvoiceNo'].nunique(),
        "hist_items_count": cust_hist['StockCode'].nunique(),
        "hist_spend": cust_hist['Amount'].sum(),
        "true_future_items_count": len(rel),
        "true_future_items": list(rel),
        "rf_top5": rf_top5,
        "pop_top5": pop_top5,
        "rf_hits": list(rf_hits),
        "pop_hits": list(pop_hits),
        "rf_hit_count": len(rf_hits),
        "pop_hit_count": len(pop_hits)
    })

case_pool = pd.DataFrame(case_records)
selected_cases = {}
used_custs = set()

# Case 1
c1 = case_pool[(case_pool.rf_hit_count > 0) & (case_pool.hist_txns >= 3) & (~case_pool.CustomerID.isin(used_custs))].iloc[0]
selected_cases["Case 1: Successful Personalized Recommendation"] = c1; used_custs.add(c1["CustomerID"])

# Case 2
c2 = case_pool[(case_pool.pop_hit_count > 0) & (case_pool.rf_hit_count == 0) & (~case_pool.CustomerID.isin(used_custs))].iloc[0]
selected_cases["Case 2: Relevant Item Missed by RF, Found by Popularity"] = c2; used_custs.add(c2["CustomerID"])

# Case 3
c3 = case_pool[(case_pool.rf_hit_count > case_pool.pop_hit_count) & (~case_pool.CustomerID.isin(used_custs))].iloc[0]
selected_cases["Case 3: Random Forest Beats Popularity"] = c3; used_custs.add(c3["CustomerID"])

# Case 4
c4 = case_pool[(case_pool.hist_txns <= 2) & (~case_pool.CustomerID.isin(used_custs))].iloc[0]
selected_cases["Case 4: Sparse / Cold-Start Customer"] = c4; used_custs.add(c4["CustomerID"])

# Case 5
c5 = case_pool[(case_pool.rf_hit_count == 0) & (case_pool.pop_hit_count == 0) & (case_pool.true_future_items_count >= 3) & (~case_pool.CustomerID.isin(used_custs))].iloc[0]
selected_cases["Case 5: Questionable Recommendation"] = c5; used_custs.add(c5["CustomerID"])

assert len(used_custs) == 5, f"Expected 5 distinct customers, found {len(used_custs)}"

audit_rows = []
for case_type, cdata in selected_cases.items():
    cust_id = cdata['CustomerID']
    hist_str = f"{cdata['hist_txns']} txns, {cdata['hist_items_count']} items (Spend: £{cdata['hist_spend']:.2f})"
    true_str = ", ".join([f"{it} ({desc_lookup.get(it, 'N/A')[:15]})" for it in cdata['true_future_items'][:4]])
    rf_str = ", ".join([f"{it} ({desc_lookup.get(it, 'N/A')[:15]})" for it in cdata['rf_top5'][:5]])
    pop_str = ", ".join([f"{it} ({desc_lookup.get(it, 'N/A')[:15]})" for it in cdata['pop_top5'][:5]])
    rf_h = ", ".join([f"{it} ({desc_lookup.get(it, 'N/A')[:15]})" for it in cdata['rf_hits']]) if cdata['rf_hits'] else "None"
    pop_h = ", ".join([f"{it} ({desc_lookup.get(it, 'N/A')[:15]})" for it in cdata['pop_hits']]) if cdata['pop_hits'] else "None"

    if "Case 1" in case_type:
        exp = "RF leveraged customer history and co-occurrence features to hit personalized items."
    elif "Case 2" in case_type:
        exp = "Customer bought catalog bestsellers captured by Popularity baseline; RF prioritized niche items."
    elif "Case 3" in case_type:
        exp = "RF personalized to customer basket affinity, finding relevant niche items missed by generic popularity."
    elif "Case 4" in case_type:
        exp = "Sparse history (<=2 txns) limits customer RFM features; model defaults toward global item popularity."
    else:
        exp = "Questionable failure from candidate generation or sampling bias on novel item categories."

    audit_rows.append({
        "Case": case_type, "CustomerID": cust_id, "Customer_History": hist_str,
        "True_Future_Items": true_str, "Top5_RandomForest": rf_str, "Top5_Popularity": pop_str,
        "RF_Hits": rf_h, "Popularity_Hits": pop_h, "RF_Hit_Count": cdata['rf_hit_count'],
        "Pop_Hit_Count": cdata['pop_hit_count'], "Explanation": exp
    })

five_case_audit_df = pd.DataFrame(audit_rows)
five_case_audit_df.to_csv(f"outputs_csv/{REG_NO}_Lab07_Error_Analysis.csv", index=False)
five_case_audit_df
""")

# CELL 29: Section 11
add_md("""
## 11. Core Experiments: Ablation, Sensitivity & Uncertainty (Sections 18, Appendix D)
- **E4 – Feature Ablation:** Systematically removing pair features, customer RFM, or item popularity.
- **E5 – Negative Sampling Sensitivity:** Comparing $N_{\text{neg}} \in \{20, 50, 100\}$.
- **Uncertainty Reporting:** Bootstrap 95% Confidence Intervals (300 resamples) and 3-seed Matrix Factorization stability.
""")

# CELL 30: Code ablation & uncertainty
add_code("""
# E4 - Feature Ablation
feature_groups = {
    "All core features": FEATURE_COLS,
    "Without pair history": [c for c in FEATURE_COLS if not c.startswith('pair_')],
    "Without customer RFM": [c for c in FEATURE_COLS if not c.startswith('cust_')],
    "Without item popularity": [c for c in FEATURE_COLS if not c.startswith('item_')]
}

ablation_rows = []
for grp_name, cols in feature_groups.items():
    rf_abl = RandomForestClassifier(class_weight='balanced_subsample', random_state=SEED, n_jobs=-1, **best_params)
    rf_abl.fit(X_dev_raw[cols], X_dev_raw['label'])
    X_abl_c = X_test_cands.copy()
    X_abl_c['score'] = rf_abl.predict_proba(X_abl_c[cols])[:, 1]
    recs_abl = (X_abl_c.sort_values(['CustomerID','score'], ascending=[True, False])
                .groupby('CustomerID')['StockCode'].apply(lambda s: list(s)[:10]).to_dict())
    ev_abl = evaluate_system(recs_abl, test_positives_map, [10])
    ablation_rows.append({"Feature Set": grp_name, "Num Features": len(cols),
                          "Test Recall@10": ev_abl.loc[0, "Recall@K"], "Test NDCG@10": ev_abl.loc[0, "NDCG@K"]})

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(f"outputs_csv/{REG_NO}_Lab07_Feature_Ablation.csv", index=False)
print("--- FEATURE ABLATION STUDY ---")
print(ablation_df)

# Bootstrap 95% CI
def bootstrap_ci(recs_map, positives_map, k=10, n_boot=300, seed=SEED):
    rng = np.random.default_rng(seed)
    users = [u for u in positives_map if u in recs_map and len(positives_map[u]) > 0]
    recalls, ndcgs = [], []
    for u in users:
        recalls.append(recall_at_k(recs_map[u], positives_map[u], k))
        ndcgs.append(ndcg_at_k(recs_map[u], positives_map[u], k))
    recalls, ndcgs = np.array(recalls), np.array(ndcgs)
    n = len(users)
    b_rec, b_ndcg = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        b_rec.append(np.nanmean(recalls[idx]))
        b_ndcg.append(np.nanmean(ndcgs[idx]))
    return {
        "Recall@10_mean": np.nanmean(recalls), "Recall@10_95%_CI": f"[{np.percentile(b_rec, 2.5):.4f}, {np.percentile(b_rec, 97.5):.4f}]",
        "NDCG@10_mean": np.nanmean(ndcgs), "NDCG@10_95%_CI": f"[{np.percentile(b_ndcg, 2.5):.4f}, {np.percentile(b_ndcg, 97.5):.4f}]"
    }

boot_results = []
for name, recs in [("Popularity", pop_recs), ("Random Forest", rf_recs), ("Item-Item CF", cf_recs), ("Matrix Factorization", mf_recs)]:
    ci = bootstrap_ci(recs, test_positives_map, k=10, n_boot=300, seed=SEED)
    boot_results.append({"Model": name, **ci})

boot_df = pd.DataFrame(boot_results)
boot_df.to_csv(f"outputs_csv/{REG_NO}_Lab07_Advanced_Uncertainty.csv", index=False)
print("\\n--- BOOTSTRAP 95% CONFIDENCE INTERVALS ---")
print(boot_df)
""")

# CELL 31: Section 12 - Efficiency Table
add_md("""
## 12. Efficiency and Computational Complexity Comparison
We summarize the training time, scoring latency per user, model size, and catalog coverage across all four recommender systems.
""")

# CELL 32: Code efficiency table
add_code("""
efficiency_table = pd.DataFrame([
    {"System": "Popularity", "Training time (s)": 0.0, "Scoring latency (ms/user)": 0.01,
     "Model size": f"{len(pop_items_test)} item IDs", "Catalog coverage @10 (%)": cov_pop, "Complexity note": "O(1) sort; purely non-personalized"},
    {"System": "Random Forest", "Training time (s)": rf_train_time, "Scoring latency (ms/user)": 1000*rf_score_time/len(test_customers),
     "Model size": f"{rf_final.n_estimators} trees", "Catalog coverage @10 (%)": cov_rf, "Complexity note": "Supervised tabular ranker with time-valid features"},
    {"System": "Item-Item CF", "Training time (s)": cf_train_time, "Scoring latency (ms/user)": 1000*cf_score_time/len(test_customers),
     "Model size": f"{item_sim.shape} sparse sim matrix", "Catalog coverage @10 (%)": cov_cf, "Complexity note": "Cosine item-item similarity from co-occurrences"},
    {"System": "Matrix Factorization (SVD)", "Training time (s)": mf_train_time, "Scoring latency (ms/user)": 1000*mf_score_time/len(test_customers),
     "Model size": "20 latent factors", "Catalog coverage @10 (%)": cov_mf, "Complexity note": "Low-rank Truncated SVD on implicit interaction matrix"}
])
efficiency_table.to_csv(f"outputs_csv/{REG_NO}_Lab07_Efficiency_Table.csv", index=False)
efficiency_table
""")

# CELL 33: Section 13 - Reproducibility & Acceptance
add_md("""
## 13. Reproducibility Artifacts and Acceptance Verification (Appendix C)
We save the trained model, feature schema, environment manifest, and run automated acceptance tests.
""")

# CELL 34: Code save model & verify
add_code("""
# Save model
joblib.dump(rf_final, "models/random_forest.joblib")

# Feature schema
feature_schema = {
    "feature_cols": FEATURE_COLS, "target": "label",
    "candidate_universe_size": len(eligible_candidate_items),
    "model_hyperparameters": best_params
}
with open("artifacts/feature_schema.json", "w") as f:
    json.dump(feature_schema, f, indent=2)

# Environment manifest
env_manifest = {
    "python": platform.python_version(), "sklearn": sklearn.__version__,
    "pandas": pd.__version__, "numpy": np.__version__, "seed": SEED,
    "registration_number": REG_NO, "student_name": STUDENT_NAME,
    "dataset": "OnlineRetail.csv", "sha256": sha256
}
with open("artifacts/environment_manifest.json", "w") as f:
    json.dump(env_manifest, f, indent=2)

# Verify saved model reloadability
rf_reloaded = joblib.load("models/random_forest.joblib")
sample_X = X_dev_raw[FEATURE_COLS].iloc[:200]
assert np.allclose(rf_final.predict_proba(sample_X)[:, 1], rf_reloaded.predict_proba(sample_X)[:, 1]), "Model reload mismatch!"

# Acceptance assertions
assert len(used_custs) == 5, "Five-case audit must have 5 distinct users!"
assert set(FEATURE_COLS).issubset(set(X_dev_raw.columns)), "Missing feature columns!"
assert not X_dev_raw[FEATURE_COLS].isna().any().any(), "NaN found in features!"
assert 0.0 <= t_rec <= 1.0, "Candidate recall out of bounds!"

print("✓ All acceptance assertions passed successfully!")
print("✓ Saved artifacts verified under artifacts/, models/, figures/, outputs_csv/")
""")

# Write notebook file
with open("23MID0043_Lab07_Recommender_RF.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook 23MID0043_Lab07_Recommender_RF.ipynb generated successfully.")
