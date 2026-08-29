import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 22)

df = pd.read_csv("jaipur_student_housing_clustered.csv")
print(f"Loaded {len(df)} clustered accommodations.\n")

AMENITY_COLUMNS = ["wifi", "ac", "food_mess", "laundry"]
AMENITY_DISPLAY_NAMES = {"wifi": "Wi-Fi", "ac": "AC", "food_mess": "Food/Mess", "laundry": "Laundry"}
# Friendly aliases a user might type, mapped to the real column names.
AMENITY_ALIASES = {
    "wifi": "wifi", "wi-fi": "wifi",
    "ac": "ac", "aircon": "ac", "airconditioning": "ac",
    "food": "food_mess", "mess": "food_mess", "food_mess": "food_mess",
    "laundry": "laundry",
}


def amenities_string(row):
    """Turn a row's binary amenity flags into a readable string, e.g. 'Wi-Fi, AC'."""
    present = [AMENITY_DISPLAY_NAMES[c] for c in AMENITY_COLUMNS if row[c] == 1]
    return ", ".join(present) if present else "None"

def find_closest_cluster(data, max_budget, max_distance_km, desired_amenity_count):
    """
    When hard filtering returns zero results, don't just say "no matches" —
    figure out which cluster's AVERAGE profile is closest to what the
    student asked for, and recommend the best listings from that cluster
    instead. This reuses the K-Means clustering itself to be genuinely
    useful in an edge case rather than just failing.
    """
    profile = data.groupby(["cluster", "cluster_label"]).agg(
        avg_rent=("monthly_rent_numeric", "mean"),
        avg_distance=("distance_to_nearest_hub_km", "mean"),
        avg_amenities=("total_amenities", "mean"),
    ).reset_index()

    # "Gap" = how far a cluster's average is from what the student wants.
    # Clipped at 0, since being BETTER than requested isn't a problem.
    profile["rent_gap"] = (profile["avg_rent"] - max_budget).clip(lower=0)
    profile["distance_gap"] = (profile["avg_distance"] - max_distance_km).clip(lower=0)
    profile["amenity_gap"] = (desired_amenity_count - profile["avg_amenities"]).clip(lower=0)

    # Normalize each gap to 0-1 so rupees, kilometres, and amenity counts
    # don't distort the comparison just because they're on different scales.
    for col in ["rent_gap", "distance_gap", "amenity_gap"]:
        col_max = profile[col].max()
        profile[col + "_norm"] = profile[col] / col_max if col_max > 0 else 0.0

    profile["total_gap"] = profile[["rent_gap_norm", "distance_gap_norm", "amenity_gap_norm"]].sum(axis=1)
    return profile.sort_values("total_gap").iloc[0]

def recommend_housing(
    data,
    max_budget,
    max_distance_km,
    required_amenities=None,
    gender_preference=None,
    accommodation_type=None,
    top_n=10,
):
    """
    Filter accommodations by hard constraints, report which cluster(s) the
    matches fall into, and return the top `top_n` matches ranked by VFM index.

    Parameters
    ----------
    data : pd.DataFrame          the clustered dataset
    max_budget : float           maximum monthly rent (Rs)
    max_distance_km : float      maximum distance from nearest hub (km)
    required_amenities : list    e.g. ['wifi', 'ac'] — column names, all must be 1
    gender_preference : str      'Boys' / 'Girls' / 'Co-ed' / None (no filter)
    accommodation_type : str     'PG' / 'Hostel' / 'Independent Flat' / None
    top_n : int                  how many results to return

    Returns
    -------
    (results_df, cluster_breakdown) — results_df is the ranked matches;
    cluster_breakdown is a Series of how many matches fell in each cluster
    (None if the fallback path was used).
    """
    required_amenities = required_amenities or []

    filtered = data.copy()
    filtered = filtered[filtered["monthly_rent_numeric"] <= max_budget]
    filtered = filtered[filtered["distance_to_nearest_hub_km"] <= max_distance_km]
    for amenity_col in required_amenities:
        filtered = filtered[filtered[amenity_col] == 1]
    if gender_preference:
        filtered = filtered[filtered["gender_preference"].str.lower() == gender_preference.lower()]
    if accommodation_type:
        filtered = filtered[filtered["accommodation_type"].str.lower() == accommodation_type.lower()]

    if filtered.empty:
        print("No accommodations satisfy every constraint exactly.")
        best_cluster = find_closest_cluster(data, max_budget, max_distance_km, len(required_amenities))
        print(
            f"Closest matching cluster: '{best_cluster['cluster_label']}' "
            f"(avg rent ~Rs.{best_cluster['avg_rent']:.0f}, "
            f"avg distance ~{best_cluster['avg_distance']:.1f} km, "
            f"avg amenities ~{best_cluster['avg_amenities']:.1f})\n"
        )
        fallback_results = (
            data[data["cluster"] == best_cluster["cluster"]]
            .sort_values("vfm_index", ascending=False)
            .head(top_n)
        )
        return fallback_results, None

    cluster_breakdown = filtered["cluster_label"].value_counts()
    print(f"Found {len(filtered)} accommodations matching your constraints.")
    print("Matches by cluster:")
    for label, count in cluster_breakdown.items():
        print(f"  {label}: {count}")
    print()

    top_matches = filtered.sort_values("vfm_index", ascending=False).head(top_n)
    return top_matches, cluster_breakdown


def display_results(results):
    """Pretty-print a results dataframe returned by recommend_housing()."""
    if results is None or results.empty:
        print("No results to display.")
        return
    view = results.copy()
    view["amenities"] = view.apply(amenities_string, axis=1)
    view["monthly_rent_numeric"] = view["monthly_rent_numeric"].map(lambda x: f"Rs.{x:.0f}")
    view["distance_to_nearest_hub_km"] = view["distance_to_nearest_hub_km"].map(lambda x: f"{x:.2f} km")
    cols = [
        "name", "accommodation_type", "gender_preference", "monthly_rent_numeric",
        "distance_to_nearest_hub_km", "nearest_hub", "amenities", "vfm_index", "cluster_label",
    ]
    print(view[cols].to_string(index=False))

def interactive_search(data):
    """Prompt the user for their preferences via the console and run the search."""
    print("=" * 70)
    print("  JAIPUR STUDENT HOUSING RECOMMENDER")
    print("=" * 70)

    try:
        max_budget = float(input("Maximum monthly budget (Rs): ").strip())
        max_distance = float(input("Maximum distance from your hub (km): ").strip())
    except ValueError:
        print("Budget and distance must be numbers. Please try again.")
        return None

    amenities_raw = input(
        "Required amenities, comma-separated (wifi, ac, food, laundry) or leave blank: "
    ).strip()
    required_amenities = []
    for token in amenities_raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key in AMENITY_ALIASES:
            required_amenities.append(AMENITY_ALIASES[key])
        else:
            print(f"  (ignoring unrecognized amenity: '{token.strip()}')")

    gender_raw = input("Preferred gender category (Boys/Girls/Co-ed) or leave blank for any: ").strip()
    type_raw = input("Preferred type (PG/Hostel/Independent Flat) or leave blank for any: ").strip()

    results, _ = recommend_housing(
        data,
        max_budget=max_budget,
        max_distance_km=max_distance,
        required_amenities=required_amenities,
        gender_preference=gender_raw or None,
        accommodation_type=type_raw or None,
    )
    print()
    display_results(results)
    return results


if __name__ == "__main__":
    interactive_search(df)
