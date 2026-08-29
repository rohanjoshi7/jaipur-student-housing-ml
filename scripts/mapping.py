"""
================================================================================
STEP 5 : INTERACTIVE FOLIUM MAP
================================================================================
Plots all clustered accommodations on an interactive map of Jaipur, color-coded
by cluster, with popups showing rent, distance, and amenities. Clusters can be
toggled on/off individually via the layer control in the top-right corner.
"""

import pandas as pd
import folium

df = pd.read_csv("data/jaipur_student_housing_clustered.csv")
print(f"Loaded {len(df)} accommodations to map.")

AMENITY_COLUMNS = ["wifi", "ac", "food_mess", "laundry"]
AMENITY_DISPLAY_NAMES = {"wifi": "Wi-Fi", "ac": "AC", "food_mess": "Food/Mess", "laundry": "Laundry"}

HUBS = {
    "MNIT Jaipur (JLN Marg)": (26.8422, 75.8130),
    "Jagatpura Education Belt": (26.8158, 75.8494),
    "Vaishali Nagar Tech Corridor": (26.9123, 75.7368),
}

# --------------------------------------------------------------------------
# 5.1  ASSIGN A FIXED COLOR TO EACH CLUSTER
# --------------------------------------------------------------------------
# A small, high-contrast palette — plenty for any reasonable K. Using hex
# colors (rather than folium's limited named-color set) means this scales
# cleanly if you rerun Phase 2 with a different K.
CLUSTER_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#bfef45"]
cluster_ids = sorted(df["cluster"].unique())
color_map = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(cluster_ids)}
label_map = df.drop_duplicates("cluster").set_index("cluster")["cluster_label"].to_dict()

# --------------------------------------------------------------------------
# 5.2  BUILD THE BASE MAP, CENTERED ON JAIPUR
# --------------------------------------------------------------------------
map_center = [df["latitude"].mean(), df["longitude"].mean()]
housing_map = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")

# --------------------------------------------------------------------------
# 5.3  MARK THE KEY HUBS THEMSELVES (distinct from housing markers)
# --------------------------------------------------------------------------
hub_group = folium.FeatureGroup(name="Key Hubs", show=True)
for hub_name, (hub_lat, hub_lon) in HUBS.items():
    folium.Marker(
        location=[hub_lat, hub_lon],
        popup=folium.Popup(f"<b>{hub_name}</b><br>Student/Tech Hub", max_width=200),
        icon=folium.Icon(color="black", icon="star", prefix="fa"),
    ).add_to(hub_group)
hub_group.add_to(housing_map)

# --------------------------------------------------------------------------
# 5.4  ADD ONE MARKER PER ACCOMMODATION, GROUPED BY CLUSTER
# --------------------------------------------------------------------------
# Each cluster gets its own FeatureGroup so the layer control can toggle
# individual clusters on/off — genuinely useful for a student comparing
# "just show me the Budget-Friendly Outskirts" vs. everything at once.
cluster_groups = {}
for cid in cluster_ids:
    label = label_map[cid]
    cluster_groups[cid] = folium.FeatureGroup(name=f"{label} (cluster {cid})", show=True)

for _, row in df.iterrows():
    present_amenities = [AMENITY_DISPLAY_NAMES[c] for c in AMENITY_COLUMNS if row[c] == 1]
    amenities_text = ", ".join(present_amenities) if present_amenities else "None"

    popup_html = f"""
    <div style="font-family: Arial, sans-serif; font-size: 13px; width: 210px;">
        <b>{row['name']}</b><br>
        {row['accommodation_type']} &middot; {row['gender_preference']}<br>
        <hr style="margin:4px 0;">
        Rent: <b>Rs.{row['monthly_rent_numeric']:.0f}/month</b><br>
        Distance: <b>{row['distance_to_nearest_hub_km']:.2f} km</b> to {row['nearest_hub']}<br>
        Amenities: {amenities_text}<br>
        VFM Index: <b>{row['vfm_index']:.1f}</b><br>
        <span style="color:{color_map[row['cluster']]};">&#9679;</span> {row['cluster_label']}
    </div>
    """

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=color_map[row["cluster"]],
        fill=True,
        fill_color=color_map[row["cluster"]],
        fill_opacity=0.75,
        weight=1,
        popup=folium.Popup(popup_html, max_width=250),
    ).add_to(cluster_groups[row["cluster"]])

for group in cluster_groups.values():
    group.add_to(housing_map)

# --------------------------------------------------------------------------
# 5.5  LAYER CONTROL (toggle clusters/hubs on and off)
# --------------------------------------------------------------------------
folium.LayerControl(collapsed=False).add_to(housing_map)

# --------------------------------------------------------------------------
# 5.6  CUSTOM LEGEND (folium has no built-in legend, so add a small HTML box)
# --------------------------------------------------------------------------
legend_rows = "".join(
    f'<div style="margin:2px 0;"><span style="display:inline-block;width:10px;height:10px;'
    f'background:{color_map[cid]};border-radius:50%;margin-right:6px;"></span>{label_map[cid]}</div>'
    for cid in cluster_ids
)
legend_html = f"""
<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
            background: white; padding: 10px 14px; border: 1px solid #999;
            border-radius: 6px; font-family: Arial, sans-serif; font-size: 12px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
    <b>Cluster Legend</b><br>{legend_rows}
</div>
"""
housing_map.get_root().html.add_child(folium.Element(legend_html))

# --------------------------------------------------------------------------
# 5.7  SAVE
# --------------------------------------------------------------------------
output_path = "jaipur_housing_map.html"
housing_map.save(output_path)
print(f"Saved interactive map -> {output_path}")
