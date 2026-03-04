# 01_data_loading.py

import pyodbc
import pandas as pd


def load_data_from_sql():
    """
    Connects to SQL Server and loads feature engineered data
    """

    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=.\SQLEXPRESS;"          # change this
        "DATABASE=predictive_maintenance_db;"      # change this
        "Trusted_Connection=yes;"
    )

    query = "SELECT * FROM vw_feature_engineered_data"

    df = pd.read_sql(query, conn)

    conn.close()

    return df


if __name__ == "__main__":
    df = load_data_from_sql()

    print("Data Loaded Successfully ✅")
    print("Shape:", df.shape)
    print(df.head())
