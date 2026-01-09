#!/usr/bin/env python3
"""Docstring for 14-visualize.py."""
import matplotlib.pyplot as plt


import pandas as pd


from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


def visualize(df):
    """Docstring for visualize."""
    df = df.drop(columns=["Weighted_Price"])

    df = df.rename(columns={"Timestamp": "Date"})

    df["Date"] = pd.to_datetime(df["Date"], unit="s")

    df = df.set_index("Date").sort_index()

    df["Close"] = df["Close"].ffill()

    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    df = df.loc["2017-01-01":]

    df_daily = df.resample("D").agg({
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    })

    return df_daily


df = visualize(df)

print(df)

df.plot()
plt.show()
