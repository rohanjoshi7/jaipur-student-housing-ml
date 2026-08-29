import numpy as np
import pandas as pd

# Reproducibility: always seed your RNG so results are identical on re-run.
# Whoever grades this (or you, six months from now) should get your exact numbers.
np.random.seed(42)

N_ACCOMMODATIONS = 500

# --------------------------------------------------------------------------
# 1.1  KEY STUDENT / TECH HUBS IN JAIPUR
# --------------------------------------------------------------------------
# These act as "anchor points" — the places students actually care about
# being close to. Coordinates are approximate (fine for a mock/demo dataset,
# not survey-grade GPS).
HUBS = {
    "MNIT Jaipur (JLN Marg)":       (26.8422, 75.8130),
    "Jagatpura Education Belt":     (26.8158, 75.8494),  # JECRC, Poornima, Arya College etc.
    "Vaishali Nagar Tech Corridor": (26.9123, 75.7368),  # World Trade Park + IT/commercial offices
}
HUB_NAMES = list(HUBS.keys())

# Rough bounding box of Jaipur city, used for the small fraction of listings
# scattered generally around town rather than tightly around one hub —
# every city has plenty of housing that isn't hub-adjacent at all.
JAIPUR_LAT_RANGE = (26.75, 26.97)
JAIPUR_LON_RANGE = (75.68, 75.90)

KM_PER_DEG_LAT = 111.0  # constant everywhere on Earth (approx)


def offset_point(hub_lat, hub_lon, distance_km, bearing_deg):
    """
    Given a hub location, a distance (km), and a bearing (degrees),
    return the (lat, lon) of a point that far away in that direction.
    Simple flat-Earth approximation — accurate enough at city scale.
    """
    km_per_deg_lon = 111.0 * np.cos(np.radians(hub_lat))  # shrinks away from equator
    bearing_rad = np.radians(bearing_deg)
    d_lat = (distance_km * np.cos(bearing_rad)) / KM_PER_DEG_LAT
    d_lon = (distance_km * np.sin(bearing_rad)) / km_per_deg_lon
    return hub_lat + d_lat, hub_lon + d_lon


# --------------------------------------------------------------------------
# 1.2  GENERATE LOCATIONS
# --------------------------------------------------------------------------
# 85% of listings cluster (with realistic decay — lots close by, fewer far
# away) around one of the three hubs. 15% are scattered generally around
# the city, representing ordinary housing with no hub in particular nearby.
near_hub_mask = np.random.rand(N_ACCOMMODATIONS) < 0.85
assigned_hub = np.random.choice(HUB_NAMES, size=N_ACCOMMODATIONS)

latitudes, longitudes = [], []
# We also track the TRUE distance-to-assigned-hub internally (not saved to
# the CSV) purely to make rent/amenities correlate realistically with
# location, mimicking real housing markets where "closer = pricier".
true_hub_distance_km = []

for i in range(N_ACCOMMODATIONS):
    if near_hub_mask[i]:
        hub_lat, hub_lon = HUBS[assigned_hub[i]]
        # Exponential distribution: many listings very close, a long tail further out.
        dist_km = np.random.exponential(scale=2.5)
        dist_km = min(dist_km, 14)  # cap so we don't get absurd outliers
        bearing = np.random.uniform(0, 360)
        lat, lon = offset_point(hub_lat, hub_lon, dist_km, bearing)
        true_hub_distance_km.append(dist_km)
    else:
        lat = np.random.uniform(*JAIPUR_LAT_RANGE)
        lon = np.random.uniform(*JAIPUR_LON_RANGE)
        # distance from nearest hub for a randomly-placed point, for rent modelling
        dists = [
            KM_PER_DEG_LAT * np.hypot(lat - hlat, (lon - hlon) * np.cos(np.radians(hlat)))
            for hlat, hlon in HUBS.values()
        ]
        true_hub_distance_km.append(min(dists))

    latitudes.append(lat)
    longitudes.append(lon)

true_hub_distance_km = np.array(true_hub_distance_km)

# --------------------------------------------------------------------------
# 1.3  QUALITY TIER (hidden variable driving rent + amenities together)
# --------------------------------------------------------------------------
# Real housing markets aren't random noise — pricier places tend to have
# more amenities. We simulate that correlation via a hidden "tier".
quality_tier = np.random.choice(
    ["budget", "mid", "premium"], size=N_ACCOMMODATIONS, p=[0.40, 0.40, 0.20]
)
tier_rent_range = {"budget": (4000, 7500), "mid": (7500, 11500), "premium": (11500, 19000)}
tier_amenity_prob = {"budget": 0.25, "mid": 0.55, "premium": 0.85}

# --------------------------------------------------------------------------
# 1.4  MONTHLY RENT (base tier + proximity premium + noise), formatted MESSILY
# --------------------------------------------------------------------------
raw_rent = np.zeros(N_ACCOMMODATIONS)
for i in range(N_ACCOMMODATIONS):
    low, high = tier_rent_range[quality_tier[i]]
    base = np.random.uniform(low, high)
    # Being close to a hub commands a premium (real-world effect), capped sensibly.
    proximity_premium = max(0, (6 - true_hub_distance_km[i])) * np.random.uniform(60, 160)
    noise = np.random.normal(0, 300)
    raw_rent[i] = max(2500, base + proximity_premium + noise)

raw_rent = np.round(raw_rent / 100) * 100  # round to nearest ₹100, like real listings


def messify_rent(amount):
    """Format a clean rent number the way scraped/real-world listings actually look —
    inconsistent formatting is realistic and gives us genuine cleaning work to do."""
    amount = int(amount)
    style = np.random.choice(
        ["inr_comma_month", "rs_dot", "plain_permonth", "inr_permonth", "plain_number"]
    )
    if style == "inr_comma_month":
        return f"₹{amount:,}/month"
    elif style == "rs_dot":
        return f"Rs. {amount}"
    elif style == "plain_permonth":
        return f"{amount} per month"
    elif style == "inr_permonth":
        return f"INR {amount}/month"
    else:
        return f"{amount}"


monthly_rent_str = [messify_rent(r) for r in raw_rent]

# --------------------------------------------------------------------------
# 1.5  AMENITY FLAGS (correlated with quality tier, each amenity independent)
# --------------------------------------------------------------------------
def amenity_flags(tier):
    p = tier_amenity_prob[tier]
    return {
        "wifi": int(np.random.rand() < min(p + 0.10, 0.97)),       # wifi is near-universal now
        "ac": int(np.random.rand() < p * 0.8),                      # AC lags behind other amenities
        "food_mess": int(np.random.rand() < p),
        "laundry": int(np.random.rand() < p * 0.9),
    }


amenities = [amenity_flags(t) for t in quality_tier]

# --------------------------------------------------------------------------
# 1.6  NAMES, ACCOMMODATION TYPE, GENDER PREFERENCE (realistic flavour text)
# --------------------------------------------------------------------------
adjectives = [
    "Sunrise", "Comfort", "Elite", "Cozy", "Royal", "Zenith", "Silver Oak",
    "Green Valley", "Golden Nest", "Blue Orchid", "Maple", "Crimson", "Prime",
    "Serene", "Urban", "Highland", "Sunshine", "Amber", "Ivory", "Copper Leaf",
]
nouns = [
    "Nest", "Residency", "Heights", "Enclave", "Homes", "Stay", "Address",
    "Court", "Manor", "Retreat", "Point", "Palace", "House", "Villa",
]
accommodation_types = ["PG", "Hostel", "Independent Flat"]
genders = ["Boys", "Girls", "Co-ed"]

names = [
    f"{np.random.choice(adjectives)} {np.random.choice(nouns)}" for _ in range(N_ACCOMMODATIONS)
]
acc_types = np.random.choice(accommodation_types, size=N_ACCOMMODATIONS, p=[0.5, 0.35, 0.15])
gender_pref = np.random.choice(genders, size=N_ACCOMMODATIONS, p=[0.4, 0.35, 0.25])

# --------------------------------------------------------------------------
# 1.7  ASSEMBLE DATAFRAME
# --------------------------------------------------------------------------
df = pd.DataFrame({
    "accommodation_id": [f"JPR{str(i+1).zfill(4)}" for i in range(N_ACCOMMODATIONS)],
    "name": names,
    "accommodation_type": acc_types,
    "gender_preference": gender_pref,
    "latitude": np.round(latitudes, 6),
    "longitude": np.round(longitudes, 6),
    "monthly_rent": monthly_rent_str,
    "wifi": [a["wifi"] for a in amenities],
    "ac": [a["ac"] for a in amenities],
    "food_mess": [a["food_mess"] for a in amenities],
    "laundry": [a["laundry"] for a in amenities],
})

# --------------------------------------------------------------------------
# 1.8  INJECT REAL-WORLD MESSINESS: missing values
# --------------------------------------------------------------------------
# ~3% missing rent (listing didn't specify price / "contact owner")
missing_rent_idx = np.random.choice(df.index, size=int(0.03 * N_ACCOMMODATIONS), replace=False)
df.loc[missing_rent_idx, "monthly_rent"] = np.nan

# ~2% missing values scattered across amenity columns (incomplete listings)
for col in ["wifi", "ac", "food_mess", "laundry"]:
    missing_idx = np.random.choice(df.index, size=int(0.02 * N_ACCOMMODATIONS), replace=False)
    df.loc[missing_idx, col] = np.nan

# --------------------------------------------------------------------------
# 1.9  SAVE
# --------------------------------------------------------------------------
output_path = "jaipur_student_housing_raw.csv"
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} accommodations -> {output_path}")
print("\nSample rows:")
print(df.head(8).to_string(index=False))
print("\nMissing values per column:")
print(df.isna().sum())
