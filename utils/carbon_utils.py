def estimate_co2(distance_km, factor=120):
    """
    factor = 120 grams per km for 2-wheeler delivery
    """
    return distance_km * factor
