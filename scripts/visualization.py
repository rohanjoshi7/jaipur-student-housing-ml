import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100  # on-screen preview; we save at higher dpi below

df = pd.read_csv("jaipur_student_housing_clustered.csv")
print(f"Loaded {len(df)} clustered accommodations for plotting.")

CLUSTER_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#bfef45"]
cluster_ids = sorted(df["cluster"].unique())
color_by_id = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(cluster_ids)}
label_by_id = df.drop_duplicates("cluster").set_index("cluster")["cluster_label"].to_dict()
palette = {label_by_id[cid]: color_by_id[cid] for cid in cluster_ids}

# Order the legend cheapest -> priciest, so the story reads naturally left to right.
hue_order = df.groupby("cluster_label")["monthly_rent_numeric"].mean().sort_values().index.tolist()


def plot_centroids(ax, x_col, y_col):
    """Overlay a bold 'X' at each cluster's average position — an easy visual
    anchor for a report reader to see where each group actually centers."""
    centroids = df.groupby("cluster_label")[[x_col, y_col]].mean()
    for label in hue_order:
        cx, cy = centroids.loc[label]
        ax.scatter(
            cx, cy, marker="X", s=240, color=palette[label],
            edgecolor="black", linewidth=1.5, zorder=5,
        )

fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=df, x="distance_to_nearest_hub_km", y="monthly_rent_numeric",
    hue="cluster_label", hue_order=hue_order, palette=palette,
    s=50, alpha=0.75, edgecolor="white", linewidth=0.4, ax=ax1,
)
plot_centroids(ax1, "distance_to_nearest_hub_km", "monthly_rent_numeric")
ax1.set_title("Rent vs. Distance to Nearest Hub, by Cluster", fontsize=13, fontweight="bold")
ax1.set_xlabel("Distance to Nearest Hub (km)")
ax1.set_ylabel("Monthly Rent (Rs.)")
ax1.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=9)
plt.tight_layout()
plt.savefig("rent_vs_distance_by_cluster.png", dpi=200, bbox_inches="tight")
print("Saved rent_vs_distance_by_cluster.png")
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=df, x="monthly_rent_numeric", y="vfm_index",
    hue="cluster_label", hue_order=hue_order, palette=palette,
    s=50, alpha=0.75, edgecolor="white", linewidth=0.4, ax=ax2,
)
plot_centroids(ax2, "monthly_rent_numeric", "vfm_index")
ax2.set_yscale("symlog", linthresh=2)
ax2.set_title("Value-for-Money Index vs. Rent, by Cluster", fontsize=13, fontweight="bold")
ax2.set_xlabel("Monthly Rent (Rs.)")
ax2.set_ylabel("VFM Index (symlog scale; higher = better value)")
ax2.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=9)
plt.tight_layout()
plt.savefig("vfm_vs_rent_by_cluster.png", dpi=200, bbox_inches="tight")
print("Saved vfm_vs_rent_by_cluster.png")
plt.close(fig2)

fig3, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(
    data=df, x="distance_to_nearest_hub_km", y="monthly_rent_numeric",
    hue="cluster_label", hue_order=hue_order, palette=palette,
    s=45, alpha=0.75, edgecolor="white", linewidth=0.4, ax=axA, legend=False,
)
plot_centroids(axA, "distance_to_nearest_hub_km", "monthly_rent_numeric")
axA.set_title("Rent vs. Distance to Nearest Hub", fontsize=13, fontweight="bold")
axA.set_xlabel("Distance to Nearest Hub (km)")
axA.set_ylabel("Monthly Rent (Rs.)")

sns.scatterplot(
    data=df, x="monthly_rent_numeric", y="vfm_index",
    hue="cluster_label", hue_order=hue_order, palette=palette,
    s=45, alpha=0.75, edgecolor="white", linewidth=0.4, ax=axB, legend=False,
)
plot_centroids(axB, "monthly_rent_numeric", "vfm_index")
axB.set_yscale("symlog", linthresh=2)
axB.set_title("VFM Index vs. Rent", fontsize=13, fontweight="bold")
axB.set_xlabel("Monthly Rent (Rs.)")
axB.set_ylabel("VFM Index (symlog scale)")

# One shared legend for both panels, built manually so it isn't duplicated.
legend_handles = [
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[label],
                  markersize=9, label=label)
    for label in hue_order
]
fig3.legend(
    handles=legend_handles, title="Cluster", loc="lower center",
    ncol=min(len(hue_order), 3), bbox_to_anchor=(0.5, -0.08), fontsize=9,
)
fig3.suptitle("Student Accommodation Clusters in Jaipur", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("cluster_scatter_combined.png", dpi=200, bbox_inches="tight")
print("Saved cluster_scatter_combined.png")
plt.close(fig3)
