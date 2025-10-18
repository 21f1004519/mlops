import pandas as pd
import pytest
import os

def test_data_integrity():
    assert os.path.exists("data/raw/iris.csv"), "Data file not found!"
    df = pd.read_csv("data/raw/iris.csv")

    # Check for nulls
    assert df.isnull().sum().sum() == 0, "Data contains missing values!"

    # Check expected columns
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_cols.issubset(df.columns), "Missing expected columns!"
