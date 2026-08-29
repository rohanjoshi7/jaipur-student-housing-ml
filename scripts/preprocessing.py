import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("jaipur_student_housing_raw.csv")
print(f"Loaded {len(df)} rows.")

def clean_rent(raw_value):
    if pd.isna(raw_value):
        return np.nan
    text = str(raw_value)
    
    text = re.sub(r"(?i)rs\.?|inr|₹|per\s*month|/\s*month", "", text)
    text = text.replace(",", "").strip()
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else np.nan


df["monthly_rent_numeric"] = df["monthly_rent"].apply(clean_rent)

median_rent = df["monthly_rent_numeric"].median()
n_missing_rent = df["monthly_rent_numeric"].isna().sum()
df["monthly_rent_numeric"] = df["monthly_rent_numeric"].fillna(median_rent)
print(f"Cleaned rent column. Imputed {n_missing_rent} missing values with median = Rs.{median_rent:.0f}")

amenity_cols = ["wifi", "ac", "food_mess", "laundry"]
for col in amenity_cols:
    mode_val = df[col].mode()[0]
    n_missing = df[col].isna().sum()
    df[col] = df[col].fillna(mode_val).astype(int)
    print(f"  {col}: imputed {n_missing} missing values with mode = {int(mode_val)}")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Returns distance in km between (lat1, lon1) and (lat2, lon2)."""
    R = 6371.0  # mean radius of Earth in km
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    d_lat = lat2_r - lat1_r
    d_lon = lon2_r - lon1_r
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


HUBS = {
    "MNIT Jaipur (JLN Marg)": (26.8422, 75.8130),
    "Jagatpura Education Belt": (26.8158, 75.8494),
    "Vaishali Nagar Tech Corridor": (26.9123, 75.7368),
}

# Compute distance to EVERY hub (useful, transparent, and lets students
# filter by "near MNIT specifically" later in the recommendation function).
hub_distance_cols = []
for hub_name, (hub_lat, hub_lon) in HUBS.items():
    col_name = "dist_" + re.sub(r"[^a-z0-9]+", "_", hub_name.lower()).strip("_") + "_km"
    df[col_name] = haversine_distance(df["latitude"], df["longitude"], hub_lat, hub_lon).round(2)
    hub_distance_cols.append(col_name)

df["distance_to_nearest_hub_km"] = df[hub_distance_cols].min(axis=1).round(2)
df["nearest_hub"] = df[hub_distance_cols].idxmin(axis=1).apply(
    lambda c: [h for h in HUBS if c.startswith("dist_" + re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_"))][0]
)
print("Computed Haversine distances to all hubs + nearest-hub distance.")

df["total_amenities"] = df[amenity_cols].sum(axis=1)

# FEATURE: VALUE-FOR-MONEY (VFM) INDEX
# --------------------------------------------------------------------------
# The idea: reward accommodations that pack in a lot of amenities for
# relatively LOW rent. We can't just divide amenities by raw rent (rupee
# scale would dominate everything), so we first min-max normalize rent to
# a clean [0, 1] range, then divide.
#
# A small epsilon is added to the denominator: without it, the single
# cheapest listing in the dataset would get normalized rent = 0, and
# amenities / 0 would blow up to infinity. Epsilon keeps the index finite
# and interpretable while still rewarding low rent heavily.
rent_min, rent_max = df["monthly_rent_numeric"].min(), df["monthly_rent_numeric"].max()
df["rent_normalized"] = (df["monthly_rent_numeric"] - rent_min) / (rent_max - rent_min)

EPSILON = 0.05
df["vfm_index"] = (df["total_amenities"] / (df["rent_normalized"] + EPSILON)).round(2)
print(f"Computed VFM index (epsilon={EPSILON}). Range: {df['vfm_index'].min():.2f} - {df['vfm_index'].max():.2f}")

features_to_scale = [
    "monthly_rent_numeric",
    "distance_to_nearest_hub_km",
    "total_amenities",
    "vfm_index",
]

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[features_to_scale])
scaled_df = pd.DataFrame(
    scaled_values,
    columns=[f"{col}_scaled" for col in features_to_scale],
    index=df.index,
)
df = pd.concat([df, scaled_df], axis=1)
print("Scaled features:", features_to_scale)


output_path = "jaipur_student_housing_processed.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved processed dataset -> {output_path}")

print("\nPreview of engineered features:")
preview_cols = [
    "name", "monthly_rent_numeric", "distance_to_nearest_hub_km", "nearest_hub",
    "total_amenities", "vfm_index",
]
print(df[preview_cols].head(8).to_string(index=False))

print("\nFinal check — any remaining nulls in engineered columns?")
print(df[features_to_scale].isna().sum())
