import pandas as pd
from ortools.constraint_solver import RoutingModel, RoutingIndexManager, pywrapcp
from utils.distance_utils import haversine_distance, estimate_travel_time
from utils.carbon_utils import estimate_co2

def build_cost_matrix(df):
    n = len(df)
    matrix = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine_distance(df.loc[i, "Store_Latitude"],
                                       df.loc[i, "Store_Longitude"],
                                       df.loc[j, "Drop_Latitude"],
                                       df.loc[j, "Drop_Longitude"])

                t = estimate_travel_time(d)
                c = estimate_co2(d)

                # Multi Objective Cost (Weight Time + CO₂)
                matrix[i][j] = t * 0.6 + c * 0.4
    return matrix

def solve_route(df):
    cost_matrix = build_cost_matrix(df)
    n = len(cost_matrix)

    manager = RoutingIndexManager(n, 1, 0)
    routing = RoutingModel(manager)

    def distance_callback(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return int(cost_matrix[f][t] * 100)

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_callback))

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        search_params.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_params)

    route = []
    index = routing.Start(0)

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    route.append(manager.IndexToNode(index))
    return route
