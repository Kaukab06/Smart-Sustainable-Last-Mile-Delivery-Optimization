#!/usr/bin/env python3
"""
add_emissions.py

Reads an input dataset (Excel or CSV), estimates trip distances (prefers distance_km; if not present,
computes haversine distance from any detected lat/lon columns), applies vehicle emission factors
(g CO2e / km), apportions emissions per parcel (by weight or number of stops) and writes an
output CSV with the added columns:
  - _estimated_distance_km
  - _emission_factor_g_per_km
  - estimated_co2_g
  - estimated_co2_kg

Usage:
  python add_emissions.py /path/to/amazon_delivery.csv.xlsx /path/to/output.csv

If no args provided it will try to use "./amazon_delivery.csv.xlsx" and write to
"./amazon_delivery_with_emissions_v2.csv"
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
import re

# ---------- Configuration ----------
DEFAULT_INPUT = "amazon_delivery.csv"
DEFAULT_OUTPUT = "amazon_delivery_with_emissions_v2.csv"

EMISSION_FACTORS = {
    "small_van": 250,   # g CO2e/km
    "large_van": 350,
    "diesel_truck": 900,
    "electric_van": 60,
    "cargo_bike": 5,
    "bike": 0,
    "scooter": 20,
    "car": 210,
    "default": 300
}
# default vehicle capacity in kg used to apportion by weight if needed
DEFAULT_VEHICLE_CAPACITY_KG = 1000.0

# ---------- Utilities ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def prefix_name(col):
    s = re.sub(r'(?i)(_)?(latitude|longitude|lat|lon|long)', '', col)
    s = re.sub(r'[_\-\s]+$', '', s)
    return s.strip('_ -').strip().lower()

# ---------- Main processing ----------
def load_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        df = pd.read_excel(p)
    except Exception:
        df = pd.read_csv(p)
    return df

def detect_columns(df):
    # distance column detection
    distance_candidates = ["distance_km","distance","route_km","trip_km","kms","km"]
    distance_col = next((c for c in distance_candidates if c in df.columns), None)

    # vehicle column detection
    vehicle_candidates = ["vehicle_type","vehicle","transport_mode","mode","vehicle_type_clean"]
    vehicle_col = next((c for c in vehicle_candidates if c in df.columns), None)

    # weight/parcel columns
    weight_candidates = ["weight_kg","weight","parcel_weight_kg","package_weight","load_kg","kg"]
    weight_col = next((c for c in weight_candidates if c in df.columns), None)

    vehicle_capacity_col = next((c for c in ["vehicle_capacity_kg","vehicle_capacity","capacity_kg","capacity"] if c in df.columns), None)

    # lat/lon detection
    lat_cols = [c for c in df.columns if re.search(r'lat', c, re.I)]
    lon_cols = [c for c in df.columns if re.search(r'lon|long', c, re.I)]
    lat_prefixes = {prefix_name(c): c for c in lat_cols}
    lon_prefixes = {prefix_name(c): c for c in lon_cols}
    common_prefixes = [p for p in lat_prefixes.keys() if p in lon_prefixes.keys()]

    pairs = []
    if len(common_prefixes) >= 2:
        for i in range(len(common_prefixes)):
            for j in range(i+1, len(common_prefixes)):
                p1 = common_prefixes[i]
                p2 = common_prefixes[j]
                pairs.append((lat_prefixes[p1], lon_prefixes[p1], lat_prefixes[p2], lon_prefixes[p2]))
    return {
        "distance_col": distance_col,
        "vehicle_col": vehicle_col,
        "weight_col": weight_col,
        "vehicle_capacity_col": vehicle_capacity_col,
        "latlon_pairs": pairs
    }

def emission_factor_from_vehicle(v):
    if pd.isna(v):
        return EMISSION_FACTORS["default"]
    v = str(v).lower()
    if "electric" in v or "ev" in v:
        return EMISSION_FACTORS["electric_van"]
    if "cargo" in v and "bike" in v:
        return EMISSION_FACTORS["cargo_bike"]
    if "bike" in v and "cargo" not in v:
        return EMISSION_FACTORS["bike"]
    if "scooter" in v:
        return EMISSION_FACTORS["scooter"]
    if "van" in v and ("small" in v or "mini" in v):
        return EMISSION_FACTORS["small_van"]
    if "van" in v:
        return EMISSION_FACTORS["large_van"]
    if "truck" in v:
        return EMISSION_FACTORS["diesel_truck"]
    if "car" in v:
        return EMISSION_FACTORS["car"]
    return EMISSION_FACTORS["default"]

def estimate_distance_for_row(row, distance_col, latlon_pairs, df):
    # 1) explicit distance column
    if distance_col and not pd.isna(row.get(distance_col)):
        try:
            val = float(row.get(distance_col))
            if val > 1000:
                val_km = val / 1000.0
                if val_km < 1000:
                    return val_km
            return val
        except:
            pass
    # 2) lat/lon pair
    if latlon_pairs:
        for pair in latlon_pairs:
            try:
                lat1 = float(row.get(pair[0]))
                lon1 = float(row.get(pair[1]))
                lat2 = float(row.get(pair[2]))
                lon2 = float(row.get(pair[3]))
                if any(pd.isna(x) for x in [lat1,lon1,lat2,lon2]):
                    continue
                return haversine(lat1, lon1, lat2, lon2)
            except:
                continue
    # 3) heuristics: time -> distance (assume 30 km/h)
    for c in ["delivery_time_minutes","time_min","duration_min","duration_minutes","travel_time_min","travel_time_minutes"]:
        if c in df.columns and not pd.isna(row.get(c)):
            try:
                mins = float(row.get(c))
                return 0.5 * mins
            except:
                pass
    return np.nan

def compute_emissions(df, cols):
    distance_col = cols["distance_col"]
    vehicle_col = cols["vehicle_col"]
    weight_col = cols["weight_col"]
    vehicle_capacity_col = cols["vehicle_capacity_col"]
    latlon_pairs = cols["latlon_pairs"]

    df["_estimated_distance_km"] = df.apply(lambda r: estimate_distance_for_row(r, distance_col, latlon_pairs, df), axis=1)
    df["_emission_factor_g_per_km"] = df.apply(lambda r: emission_factor_from_vehicle(r.get(vehicle_col)) if vehicle_col else EMISSION_FACTORS["default"], axis=1)

    def compute_row_emission(r):
        d = r.get("_estimated_distance_km")
        ef = r.get("_emission_factor_g_per_km")
        if pd.isna(d) or pd.isna(ef):
            return np.nan
        base_g = ef * d
        # scale by weight share if available
        if weight_col and not pd.isna(r.get(weight_col)):
            try:
                parcel_w = float(r.get(weight_col))
                vehicle_cap = DEFAULT_VEHICLE_CAPACITY_KG
                if vehicle_capacity_col and not pd.isna(r.get(vehicle_capacity_col)):
                    try:
                        vehicle_cap = float(r.get(vehicle_capacity_col))
                    except:
                        vehicle_cap = vehicle_cap
                load_share = min(max(parcel_w / vehicle_cap, 0.01), 1.0)
                return base_g * load_share
            except:
                return base_g
        # else try apportioning by num_packages/stops
        for c in ["num_packages","packages","stops","stops_count"]:
            if c in df.columns and not pd.isna(r.get(c)):
                try:
                    n = float(r.get(c))
                    if n > 0:
                        return base_g * (1.0 / n)
                except:
                    pass
        return base_g

    df["estimated_co2_g"] = df.apply(compute_row_emission, axis=1)
    df["estimated_co2_kg"] = df["estimated_co2_g"] / 1000.0
    return df

# ---------- Entry point ----------
def main(input_path=None, output_path=None):
    in_path = input_path or DEFAULT_INPUT
    out_path = output_path or DEFAULT_OUTPUT
    print(f"Loading {in_path} ...")
    df = load_file(in_path)
    print(f"Dataset shape: {df.shape}. Columns: {list(df.columns)[:20]}")
    cols = detect_columns(df)
    print("Detected columns:", cols)
    df_out = compute_emissions(df, cols)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote output to {out_path}")
    missing_dist = int(df_out["_estimated_distance_km"].isna().sum())
    print(f"Missing distance estimates: {missing_dist} / {len(df_out)}")
    return out_path, df_out

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else None
    outp = sys.argv[2] if len(sys.argv) > 2 else None
    main(inp, outp)
