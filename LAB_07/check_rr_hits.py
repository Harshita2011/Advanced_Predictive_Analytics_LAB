import os, json, ast
import pandas as pd
import numpy as np

# Load events
events = pd.read_csv('Retailrocket/events.csv')
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

# Cutoffs used in Retailrocket notebook
t1 = pd.to_datetime("2015-08-01 00:00:00")
t2 = pd.to_datetime("2015-09-01 00:00:00")

train_hist = events[events.datetime < t1]
val_future = events[(events.datetime >= t1) & (events.datetime < t2)]
test_future = events[events.datetime >= t2]
history_for_test = events[events.datetime < t2]

# Load recommendations
recs_df = pd.read_csv('outputs_csv/23MID0043_Lab07_Retailrocket_Recommendations.csv')
recs_df['Top10_RandomForest'] = recs_df['Top10_RandomForest'].apply(ast.literal_eval)
recs_df['Top10_Popularity'] = recs_df['Top10_Popularity'].apply(ast.literal_eval)

# Test positives map (transactions or all events in test_future)
# In Retailrocket notebook:
# Target was defined on transactions/addtocart or views
print("Checking test target definition...")
test_positives_map = test_future.groupby('visitorid')['itemid'].apply(set).to_dict()

# Calculate hits for every test customer
hist_sizes = history_for_test.groupby('visitorid')['itemid'].nunique().to_dict()

rows = []
for idx, r in recs_df.iterrows():
    vis = r['visitorid']
    rel = test_positives_map.get(vis, set())
    if not rel:
        continue
    rf_top5 = r['Top10_RandomForest'][:5]
    pop_top5 = r['Top10_Popularity'][:5]
    rf_top10 = r['Top10_RandomForest'][:10]
    pop_top10 = r['Top10_Popularity'][:10]
    
    rf_hits5 = set(rf_top5) & rel
    pop_hits5 = set(pop_top5) & rel
    rf_hits10 = set(rf_top10) & rel
    pop_hits10 = set(pop_top10) & rel
    
    rows.append({
        'visitorid': vis,
        'history_size': hist_sizes.get(vis, 0),
        'true_items_count': len(rel),
        'true_items': list(rel),
        'rf_hits@5': len(rf_hits5),
        'pop_hits@5': len(pop_hits5),
        'rf_hits@10': len(rf_hits10),
        'pop_hits@10': len(pop_hits10),
        'rf_hits5_list': list(rf_hits5),
        'pop_hits5_list': list(pop_hits5),
        'rf_top5': rf_top5,
        'pop_top5': pop_top5
    })

pool_df = pd.DataFrame(rows)
print(f"Total test visitors in pool: {len(pool_df)}")
print(f"Visitors with rf_hits@5 > pop_hits@5: {len(pool_df[pool_df['rf_hits@5'] > pool_df['pop_hits@5']])}")
print(f"Visitors with rf_hits@10 > pop_hits@10: {len(pool_df[pool_df['rf_hits@10'] > pool_df['pop_hits@10']])}")
print(f"Visitors with pop_hits@5 > rf_hits@5: {len(pool_df[pool_df['pop_hits@5'] > pool_df['rf_hits@5']])}")
print(f"Visitors with rf_hits@5 > 0: {len(pool_df[pool_df['rf_hits@5'] > 0])}")

print("\n--- Visitors where rf_hits@5 > pop_hits@5 ---")
print(pool_df[pool_df['rf_hits@5'] > pool_df['pop_hits@5']].head(10)[['visitorid', 'history_size', 'true_items_count', 'rf_hits@5', 'pop_hits@5', 'rf_hits5_list']])

print("\n--- Visitors where rf_hits@10 > pop_hits@10 ---")
print(pool_df[pool_df['rf_hits@10'] > pool_df['pop_hits@10']].head(10)[['visitorid', 'history_size', 'true_items_count', 'rf_hits@10', 'pop_hits@10']])
