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

CLUSTER_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#bfef45"]
cluster_ids = sorted(df["cluster"].unique())
color_map = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(cluster_ids)}
label_map = df.drop_duplicates("cluster").set_index("cluster")["cluster_label"].to_dict()

map_center = [df["latitude"].mean(), df["longitude"].mean()]
housing_map = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")

hub_group = folium.FeatureGroup(name="Key Hubs", show=True)
for hub_name, (hub_lat, hub_lon) in HUBS.items():
    folium.Marker(
        location=[hub_lat, hub_lon],
        popup=folium.Popup(f"<b>{hub_name}</b><br>Student/Tech Hub", max_width=200),
        icon=folium.Icon(color="black", icon="star", prefix="fa"),
    ).add_to(hub_group)
hub_group.add_to(housing_map)

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

folium.LayerControl(collapsed=False).add_to(housing_map)

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

output_path = "jaipur_housing_map.html"
housing_map.save(output_path)
print(f"Saved interactive map -> {output_path}")
