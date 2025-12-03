import pandas as pd
from utils.vrp_utils import solve_route

class RouteOptimizer:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)

    def optimize(self):
        small_df = self.df.head(12)   # VRP requires smaller batches
        route = solve_route(small_df)

        optimized = small_df.iloc[route]
        return optimized
