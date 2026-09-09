import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and clean data
print("Loading OnlineRetail.csv...")
df_raw = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')
df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])

df_with_id = df_raw.dropna(subset=['CustomerID']).copy()
df_with_id['CustomerID'] = df_with_id['CustomerID'].astype(int).astype(str)

df_with_id['is_cancel'] = df_with_id['InvoiceNo'].astype(str).str.startswith('C')
df_clean = df_with_id[(~df_with_id['is_cancel']) & (df_with_id['Quantity'] > 0) & (df_with_id['UnitPrice'] > 0)].copy()
df_clean = df_clean.drop_duplicates()
df_clean['Amount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Reference cutoff date (max date + 1 second, or train cutoff)
cutoff = df_clean['InvoiceDate'].max() + pd.Timedelta(seconds=1)

# 2. Compute RFM Features
rfm = df_clean.groupby('CustomerID').agg(
    cust_txns=('InvoiceNo', 'nunique'),
    cust_spend=('Amount', 'sum'),
    cust_last=('InvoiceDate', 'max')
).reset_index()

rfm['cust_recency_days'] = (cutoff - rfm['cust_last']).dt.total_seconds() / 86400.0

print(f"Computed RFM for {len(rfm):,} unique customers.")
print(rfm[['cust_recency_days', 'cust_txns', 'cust_spend']].describe())

# 3. Create High-Quality 3 Side-by-Side Plots
sns.set_theme(style="whitegrid", font_scale=1.05)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

# Colors
color_r = '#2b6cb0' # Royal blue
color_f = '#319795' # Teal
color_m = '#805ad5' # Purple
color_ref = '#e53e3e' # Coral red

# --- Plot 1: Recency (cust_recency_days) ---
sns.histplot(
    data=rfm,
    x='cust_recency_days',
    bins=35,
    kde=True,
    color=color_r,
    edgecolor='white',
    linewidth=0.8,
    ax=axes[0]
)
med_r = rfm['cust_recency_days'].median()
axes[0].axvline(med_r, color=color_ref, linestyle='--', lw=2, label=f'Median: {med_r:.1f} days')
axes[0].set_title("Recency Distribution\n(cust_recency_days)", fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel("Days Since Last Purchase", fontsize=11, fontweight='semibold')
axes[0].set_ylabel("Number of Customers", fontsize=11, fontweight='semibold')
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

# --- Plot 2: Frequency (cust_txns) ---
sns.histplot(
    data=rfm,
    x='cust_txns',
    bins=35,
    kde=False,
    color=color_f,
    edgecolor='white',
    linewidth=0.8,
    ax=axes[1]
)
axes[1].set_yscale('log')
med_f = rfm['cust_txns'].median()
axes[1].axvline(med_f, color=color_ref, linestyle='--', lw=2, label=f'Median: {med_f:.0f} orders')
axes[1].set_title("Frequency Distribution\n(cust_txns, Log Scale)", fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel("Number of Transactions (Orders)", fontsize=11, fontweight='semibold')
axes[1].set_ylabel("Customer Count (Log Scale)", fontsize=11, fontweight='semibold')
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

# --- Plot 3: Monetary (cust_spend) ---
positive_spend = rfm[rfm['cust_spend'] > 0]['cust_spend']
sns.histplot(
    x=positive_spend,
    bins=35,
    kde=False,
    color=color_m,
    edgecolor='white',
    linewidth=0.8,
    log_scale=True,
    ax=axes[2]
)
med_m = positive_spend.median()
axes[2].axvline(med_m, color=color_ref, linestyle='--', lw=2, label=f'Median: £{med_m:,.2f}')
axes[2].set_title("Monetary Distribution\n(cust_spend, Log Scale)", fontsize=13, fontweight='bold', pad=10)
axes[2].set_xlabel("Total Spend (£, Log Scale)", fontsize=11, fontweight='semibold')
axes[2].set_ylabel("Number of Customers", fontsize=11, fontweight='semibold')
axes[2].legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

# Super Title & Tight Layout
plt.suptitle("Figure 4. Customer Recency, Frequency and Monetary (RFM) Distributions", 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

# Ensure destination folders exist
os.makedirs("figures", exist_ok=True)
out_path_1 = "figures/04_rfm_distributions.png"
out_path_2 = r"C:\Users\harsh\.gemini\antigravity-ide\brain\7bee230e-4b01-469e-a66c-b07ecb0090d1\figure4_rfm_distributions.png"

plt.savefig(out_path_1, dpi=300, bbox_inches='tight')
plt.savefig(out_path_2, dpi=300, bbox_inches='tight')
print(f"Saved figure to {out_path_1}")
print(f"Saved figure to {out_path_2}")
