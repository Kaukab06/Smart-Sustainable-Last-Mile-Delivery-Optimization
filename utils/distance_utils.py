import numpy as np
from geopy.distance import geodesic

def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def estimate_travel_time(distance_km):
    avg_speed = 30  # km/h for city traffic
    return (distance_km / avg_speed) * 60  # minutes
