import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")

df = pd.read_csv("jaipur_student_housing_processed.csv")
print(f"Loaded {len(df)} accommodations with {df.shape[1]} columns.\n")

scaled_cols = [
    "monthly_rent_numeric_scaled",
    "distance_to_nearest_hub_km_scaled",
    "total_amenities_scaled",
    "vfm_index_scaled",
]
corr_matrix = df[scaled_cols].corr().round(2)
print("Correlation matrix of scaled features (feature-selection sanity check):")
print(corr_matrix, "\n")

FEATURE_COLS = scaled_cols
X = df[FEATURE_COLS].values

K_RANGE = range(1, 11)              # inertia is defined even at K=1, so start there
SILHOUETTE_MIN_K = 2                # silhouette is undefined for a single cluster
INTERPRETABLE_MAX_K = 6             # cap the search for K used for FINAL selection —
                                     # beyond ~6 clusters, giving each an intuitive
                                     # student-friendly name stops being practical,
                                     # even if the silhouette score keeps inching up.

inertias = []
silhouette_scores = {}

for k in K_RANGE:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    inertias.append(model.inertia_)
    if k >= SILHOUETTE_MIN_K:
        silhouette_scores[k] = silhouette_score(X, labels)

candidate_ks = {k: s for k, s in silhouette_scores.items() if k <= INTERPRETABLE_MAX_K}
BEST_K = max(candidate_ks, key=candidate_ks.get)

print("Silhouette scores by K:")
for k, s in silhouette_scores.items():
    marker = "  <-- selected" if k == BEST_K else ""
    print(f"  K={k}: {s:.4f}{marker}")
print(f"\nSelected K = {BEST_K} (highest silhouette score within K <= {INTERPRETABLE_MAX_K})\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Elbow plot
sns.lineplot(x=list(K_RANGE), y=inertias, marker="o", ax=ax1, color="#4C72B0")
ax1.axvline(BEST_K, color="crimson", linestyle="--", alpha=0.7, label=f"Chosen K = {BEST_K}")
ax1.set_title("Elbow Method: Inertia vs. K", fontsize=13, fontweight="bold")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax1.set_xticks(list(K_RANGE))
ax1.legend()

# Silhouette plot
sil_ks = list(silhouette_scores.keys())
sil_vals = list(silhouette_scores.values())
sns.lineplot(x=sil_ks, y=sil_vals, marker="o", ax=ax2, color="#55A868")
ax2.axvline(BEST_K, color="crimson", linestyle="--", alpha=0.7, label=f"Chosen K = {BEST_K}")
ax2.set_title("Silhouette Score vs. K", fontsize=13, fontweight="bold")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Average Silhouette Score")
ax2.set_xticks(sil_ks)
ax2.legend()

plt.tight_layout()
plt.savefig("phase2_elbow_silhouette.png", dpi=150)
print("Saved plot -> phase2_elbow_silhouette.png")
plt.show()

# TRAINING
final_model = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df["cluster"] = final_model.fit_predict(X)

final_silhouette = silhouette_score(X, df["cluster"])
print(f"Final model trained with K={BEST_K}. Silhouette score: {final_silhouette:.4f}\n")

summary = df.groupby("cluster").agg(
    num_listings=("cluster", "count"),
    avg_rent=("monthly_rent_numeric", "mean"),
    avg_distance_km=("distance_to_nearest_hub_km", "mean"),
    avg_amenities=("total_amenities", "mean"),
    avg_vfm_index=("vfm_index", "mean"),
).round(2)

summary = summary.sort_values("avg_rent").reset_index()

rent_rank = summary["avg_rent"].rank(method="first").astype(int) - 1        # 0 = cheapest
distance_rank = summary["avg_distance_km"].rank(method="first").astype(int) - 1  # 0 = closest
vfm_rank = summary["avg_vfm_index"].rank(ascending=False, method="first").astype(int) - 1  # 0 = best value

def name_cluster(rent_r, dist_r, vfm_r, n_clusters):
    if dist_r == 0:
        return "Central Student Pockets"
    if vfm_r == 0:
        return "Best Value-for-Money Zone"
    if rent_r == 0:
        return "Budget-Friendly Outskirts"
    if rent_r == n_clusters - 1:
        return "Premium & Upscale Enclave"
    return "Balanced Mid-Range Cluster"

summary["cluster_label"] = [
    name_cluster(r, d, v, BEST_K) for r, d, v in zip(rent_rank, distance_rank, vfm_rank)
]

label_counts = summary["cluster_label"].value_counts()
duplicate_labels = label_counts[label_counts > 1].index
for lbl in duplicate_labels:
    tied = summary[summary["cluster_label"] == lbl].sort_values("avg_distance_km")
    n_tied = len(tied)
    for position, row_index in enumerate(tied.index):
        if n_tied == 2:
            suffix = " (Near Hub)" if position == 0 else " (Far from Hub)"
        else:
            suffix = f" (Group {position + 1})"
        summary.loc[row_index, "cluster_label"] = lbl + suffix

assert summary["cluster_label"].is_unique, "Cluster labels must be unique after disambiguation."

print("=" * 100)
print("CLUSTER PROFILE SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))

label_map = dict(zip(summary["cluster"], summary["cluster_label"]))
df["cluster_label"] = df["cluster"].map(label_map)

output_path = "jaipur_student_housing_clustered.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved clustered dataset -> {output_path}")
