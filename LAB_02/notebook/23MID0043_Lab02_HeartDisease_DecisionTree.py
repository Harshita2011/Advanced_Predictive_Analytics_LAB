# %% Cell 5
# Import essential Python libraries for data analysis and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import data loader and scikit-learn tools
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import joblib

# Set a clean plot style and lock random seed for reproducibility
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11
RANDOM_STATE = 42

print("Libraries imported successfully!")
print("Random seed set to:", RANDOM_STATE)

# %% Cell 7
# Load the UCI Heart Disease dataset (ID = 45)
heart_data = fetch_ucirepo(id=45)
X = heart_data.data.features.copy()

# The raw target column ('num') has values from 0 to 4.
# We convert it to a simple binary outcome:
# 0 = No heart disease, 1 = Heart disease present (values 1, 2, 3, or 4)
y = (heart_data.data.targets['num'] > 0).astype(int)
y.name = "heart_disease"

# Display dataset dimensions and class counts
print("Number of patients (rows):", len(X))
print("Number of features (columns):", len(X.columns))
print("\nDiagnosis breakdown (0 = Healthy, 1 = Heart Disease):")
print(y.value_counts().sort_index())
print(f"\nPercentage of patients with heart disease: {y.mean() * 100:.2f}%")

# %% Cell 9
# Check column data types and see if there are any duplicate patient records
print("Data types of our features:\n", X.dtypes.value_counts())
print("\nNumber of duplicate rows found:", X.duplicated().sum())

# Show summary statistics (mean, min, max, etc.) for the first few features
X.describe().T[['count', 'mean', 'std', 'min', 'max']].head(8)

# %% Cell 11
# Check how many missing values (NaN) exist in each column
missing = X.isna().sum()
print("Columns with missing values:")
print(missing[missing > 0])

# Since only 6 patients out of 303 have missing values (less than 2%),
# we can easily fill them using the median value of each column.
X_clean = X.fillna(X.median())

print("\nTotal missing values after median filling:", X_clean.isna().sum().sum())
print("Our data is now completely clean and ready for modeling!")

# %% Cell 14
# Plot bar chart and pie chart for diagnosis breakdown
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
counts = y.value_counts().sort_index()

# Bar chart
ax[0].bar(['Healthy (0)', 'Heart Disease (1)'], counts, color=['#3172af', '#d95f02'], width=0.45, edgecolor='black')
ax[0].set_title('Patient Count by Diagnosis', fontweight='bold'); ax[0].set_ylabel('Number of Patients'); ax[0].set_ylim(0, 200)
for i, val in enumerate(counts): ax[0].text(i, val + 4, str(val), ha='center', fontweight='bold')

# Pie chart
ax[1].pie(counts, labels=['Healthy (0)', 'Heart Disease (1)'], autopct='%1.1f%%', colors=['#3172af', '#d95f02'], startangle=140, wedgeprops=dict(edgecolor='black'))
ax[1].set_title('Percentage Breakdown', fontweight='bold')

plt.tight_layout(); plt.savefig('results/heart_disease_eda_class_distribution.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 16
# Plot histograms comparing healthy patients vs heart disease patients
features_to_plot = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
fig, axes = plt.subplots(2, 3, figsize=(15, 9)); axes = axes.flatten()

df_temp = X.assign(Diagnosis=y.map({0: 'Healthy', 1: 'Heart Disease'}))
for i, col in enumerate(features_to_plot):
    sns.histplot(data=df_temp, x=col, hue='Diagnosis', kde=True, element='step', palette=['#3172af', '#d95f02'], ax=axes[i], alpha=0.5, bins=18)
    axes[i].set_title(f'Distribution of {col}', fontweight='bold'); axes[i].set_xlabel('')

plt.tight_layout(); plt.savefig('results/heart_disease_eda_feature_distribution.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 18
# Create a simple table confirming 0 missing values remain
clean_check = pd.DataFrame({
    'Feature': X.columns,
    'Missing_Count': X.isna().sum(),
    'Percentage': (X.isna().sum() / len(X)) * 100
})
print("Missing value status across all columns:")
print(clean_check.to_string(index=False))

# %% Cell 20
# Plot correlation heatmap to see how features relate to each other
corr_matrix = X.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap", fontweight='bold', pad=12)

plt.tight_layout(); plt.savefig('results/heart_disease_eda_correlation_analysis.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 22
# Split data into 80% training set and 20% testing set
# We use stratify=y so that both sets keep the same percentage of heart disease patients
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

print("Training set size:", len(X_train), "patients | Heart disease percentage:", f"{y_train.mean()*100:.1f}%")
print("Testing set size: ", len(X_test), "patients  | Heart disease percentage:", f"{y_test.mean()*100:.1f}%")

# %% Cell 24
# Train a simple dummy classifier that makes random guesses based on class percentages
dummy_model = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
dummy_model.fit(X_train, y_train)

dummy_preds = dummy_model.predict(X_test)
dummy_probs = dummy_model.predict_proba(X_test)[:, 1]

print("--- Baseline Dummy Model Performance ---")
print("Accuracy:       ", round(accuracy_score(y_test, dummy_preds), 4))
print("Recall:         ", round(recall_score(y_test, dummy_preds), 4))
print("ROC-AUC Score:  ", round(roc_auc_score(y_test, dummy_probs), 4))

# Save the baseline model artifact
joblib.dump(dummy_model, 'models/heart_disease_baseline.joblib')

# %% Cell 26
# Train a standard Decision Tree Classifier without limits on tree depth
cart_model = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced')
cart_model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, cart_model.predict(X_train))
test_acc = accuracy_score(y_test, cart_model.predict(X_test))
test_rec = recall_score(y_test, cart_model.predict(X_test))
test_auc = roc_auc_score(y_test, cart_model.predict_proba(X_test)[:, 1])

print("--- Basic Decision Tree Results ---")
print("Tree Depth:     ", cart_model.get_depth(), "| Number of Leaves:", cart_model.get_n_leaves())
print("Training Accuracy:", round(train_acc, 4), "(100% means the tree memorized every patient!)")
print("Testing Accuracy: ", round(test_acc, 4))
print("Testing Recall:   ", round(test_rec, 4))
print("Testing ROC-AUC:  ", round(test_auc, 4))
joblib.dump(cart_model, 'models/heart_disease_basic_cart.joblib')

# Plot the first 3 levels of the tree
fig, ax = plt.subplots(figsize=(16, 9))
plot_tree(cart_model, feature_names=X.columns, class_names=['Healthy', 'Heart Disease'], filled=True, rounded=True, max_depth=3, fontsize=9, ax=ax)
ax.set_title("Basic Decision Tree (Truncated at Depth=3)", fontweight='bold')
plt.tight_layout(); plt.savefig('results/heart_disease_basic_cart_tree.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 28
# Use GridSearchCV to test different tree depths and pruning parameters (ccp_alpha)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'ccp_alpha': [0.0, 0.005, 0.01, 0.02]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
tuned_model = grid_search.best_estimator_

tuned_preds = tuned_model.predict(X_test)
tuned_probs = tuned_model.predict_proba(X_test)[:, 1]

print("--- Tuned & Pruned Tree Results ---")
print("Best Hyperparameters:", grid_search.best_params_)
print("Pruned Tree Depth:  ", tuned_model.get_depth(), "| Number of Leaves:", tuned_model.get_n_leaves())
print("Testing Accuracy:   ", round(accuracy_score(y_test, tuned_preds), 4))
print("Testing Recall:     ", round(recall_score(y_test, tuned_preds), 4))
print("Testing ROC-AUC:    ", round(roc_auc_score(y_test, tuned_probs), 4))
joblib.dump(tuned_model, 'models/heart_disease_tuned_cart.joblib')
pd.DataFrame(grid_search.cv_results_).to_csv('results/heart_disease_cv_results.csv', index=False)

# Plot the clean, pruned tree structure
fig, ax = plt.subplots(figsize=(15, 7))
plot_tree(tuned_model, feature_names=X.columns, class_names=['Healthy', 'Heart Disease'], filled=True, rounded=True, fontsize=10, ax=ax)
ax.set_title("Tuned & Pruned Decision Tree", fontweight='bold')
plt.tight_layout(); plt.savefig('results/heart_disease_tuned_pruned_tree.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 30
# Train a Random Forest model combining 200 individual decision trees
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=6, min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced', oob_score=True, random_state=RANDOM_STATE, n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("--- Random Forest Ensemble Performance ---")
print("Number of Trees:       ", rf_model.n_estimators)
print("Out-of-Bag (OOB) Score:", round(rf_model.oob_score_, 4))
print("Testing Accuracy:      ", round(accuracy_score(y_test, rf_preds), 4))
print("Testing Recall:        ", round(recall_score(y_test, rf_preds), 4))
print("Testing ROC-AUC Score: ", round(roc_auc_score(y_test, rf_probs), 4))
joblib.dump(rf_model, 'models/heart_disease_random_forest.joblib')

# %% Cell 33
# Plot 2x2 grid of confusion matrices for all 4 models
fig, axes = plt.subplots(2, 2, figsize=(13, 11)); axes = axes.flatten()
for i, (name, (preds, _)) in enumerate(models_dict.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i], xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'], annot_kws={'size': 14, 'weight': 'bold'})
    axes[i].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, preds):.3f} | Recall: {recall_score(y_test, preds):.3f}', fontweight='bold')
    axes[i].set_ylabel('Actual Diagnosis'); axes[i].set_xlabel('Model Prediction')
plt.tight_layout(); plt.savefig('results/heart_disease_eval_confusion_matrices.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 35
# Plot ROC Curves to compare model discrimination power
fig, ax = plt.subplots(figsize=(9, 7))
for (name, (_, probs)), col in zip(models_dict.items(), colors_list):
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, probs):.3f})', color=col, linewidth=2.5)
ax.plot([0, 1], [0, 1], 'k--', label='Chance Guess'); ax.set_title("ROC Curves Comparison", fontweight='bold'); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)"); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('results/heart_disease_eval_roc_curves.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 37
# Plot Precision-Recall Curves focusing on the heart disease class
fig, ax = plt.subplots(figsize=(9, 7))
for (name, (_, probs)), col in zip(models_dict.items(), colors_list):
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ax.plot(rec, prec, label=f'{name} (AP = {average_precision_score(y_test, probs):.3f})', color=col, linewidth=2.5)
ax.axhline(y=y_test.mean(), color='k', linestyle='--', label='Baseline'); ax.set_title("Precision–Recall Curves", fontweight='bold'); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(loc='lower left')
plt.tight_layout(); plt.savefig('results/heart_disease_eval_pr_curves.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 39
# Plot Probability Calibration to see if predicted probabilities match real outcomes
fig, ax = plt.subplots(figsize=(9, 7)); ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", linewidth=2)
for (name, (_, probs)), col in zip(models_dict.items(), colors_list):
    if 'Dummy' in name: continue
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=5, strategy='uniform')
    ax.plot(prob_pred, prob_true, "s-", label=f"{name} (Brier = {brier_score_loss(y_test, probs):.3f})", color=col, linewidth=2)
ax.set_title("Probability Calibration Chart", fontweight='bold'); ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Actual Percentage"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig('results/heart_disease_eval_calibration_plots.png', dpi=300, bbox_inches='tight'); plt.show()

# %% Cell 41
# Compare feature importances using Gini importance and Permutation importance
gini_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
perm_res = permutation_importance(rf_model, X_test, y_test, n_repeats=15, random_state=RANDOM_STATE, scoring='balanced_accuracy')
perm_imp = pd.Series(perm_res.importances_mean, index=X.columns)

imp_df = pd.DataFrame({'Gini_Importance': gini_imp, 'Permutation_Importance': perm_imp}).sort_values(by='Permutation_Importance', ascending=False)
imp_df.to_csv('results/heart_disease_feature_importance.csv')

top10 = imp_df.head(10).sort_values(by='Permutation_Importance', ascending=True)
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].barh(top10.index, top10['Gini_Importance'], color='#3172af', edgecolor='black'); ax[0].set_title('Top 10 (Gini Importance)', fontweight='bold')
ax[1].barh(top10.index, top10['Permutation_Importance'], color='#d95f02', edgecolor='black'); ax[1].set_title('Top 10 (Permutation Importance)', fontweight='bold')
plt.tight_layout(); plt.savefig('results/heart_disease_feature_importance.png', dpi=300, bbox_inches='tight'); plt.show()
imp_df.head(10).round(4)

# %% Cell 43
# Create a summary table comparing all our models across key metrics
records = []
for name, (preds, probs) in models_dict.items():
    records.append({
        'Model Name': name,
        'Accuracy': round(accuracy_score(y_test, preds), 4),
        'Recall (Sensitivity)': round(recall_score(y_test, preds), 4),
        'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
        'F1-Score': round(f1_score(y_test, preds, zero_division=0), 4),
        'ROC-AUC Score': round(roc_auc_score(y_test, probs), 4),
        'Brier Loss Score': round(brier_score_loss(y_test, probs), 4)
    })
comp_df = pd.DataFrame(records)
comp_df.to_csv('results/heart_disease_model_comparison.csv', index=False)

# Save our best model artifact for future use
joblib.dump(rf_model, 'models/heart_disease_best_model.joblib')
joblib.dump(rf_model, 'models/23MID0043_Lab02_HeartDisease_Model.joblib')
comp_df

# %% Cell 45
# Examine where our model made mistakes on the test set
test_err_df = X_test.copy()
test_err_df['Actual'] = y_test; test_err_df['Predicted'] = rf_preds

def classify_error(row):
    if row['Actual'] == 1 and row['Predicted'] == 1: return 'True Positive (TP)'
    if row['Actual'] == 0 and row['Predicted'] == 0: return 'True Negative (TN)'
    if row['Actual'] == 1 and row['Predicted'] == 0: return 'False Negative (FN)'
    if row['Actual'] == 0 and row['Predicted'] == 1: return 'False Positive (FP)'

test_err_df['Result_Type'] = test_err_df.apply(classify_error, axis=1)
err_summary = test_err_df['Result_Type'].value_counts()
print("Test Set Breakdown by Prediction Type:\n", err_summary)
err_summary.reset_index().to_csv('results/heart_disease_error_analysis.csv', index=False)

# Plot boxplots to see feature values for correct predictions vs mistakes
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.boxplot(data=test_err_df, x='Result_Type', y='oldpeak', ax=axes[0], color='#3172af'); axes[0].set_title('ST Depression by Outcome', fontweight='bold'); axes[0].set_xlabel('')
sns.boxplot(data=test_err_df, x='Result_Type', y='thalach', ax=axes[1], color='#d95f02'); axes[1].set_title('Max Heart Rate by Outcome', fontweight='bold'); axes[1].set_xlabel('')
plt.tight_layout(); plt.savefig('results/heart_disease_error_analysis_boxplots.png', dpi=300, bbox_inches='tight'); plt.show()

