import pandas as pd

def load_amazon(path):
    df = pd.read_excel("C:/Users/HP/OneDrive/Documents/amazon_delivery.xlsx")

    print("Amazon data loaded:", df.shape)
    return df

def load_emissions(path):
    df = pd.read_csv("C:/Users/HP/Downloads/CO2_Emissions_Vehicles.csv")
    print("Emission data loaded:", df.shape)
    return df

def merge_data(amazon_df, emission_df):
    df = pd.merge(amazon_df, emission_df,
                  how="left",
                  left_on="Type_of_vehicle",
                  right_on="Vehicle")
    print("Merged data shape:", df.shape)
    return df