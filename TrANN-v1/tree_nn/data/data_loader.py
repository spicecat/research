"""Data loading and preprocessing utilities."""

import io
import ssl
import urllib.request

import numpy as np
import pandas as pd


class DataLoader:
    @staticmethod
    def get_data(url):
        """
        Safely download data from URL with SSL verification handling.
        """
        try:
            return pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
        except urllib.error.URLError:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(url, context=ssl_context) as response:
                data = response.read()
                return pd.read_csv(
                    io.StringIO(data.decode("utf-8")),
                    sep=r"\s+",
                    skiprows=22,
                    header=None,
                )

    @staticmethod
    def load_boston_data():
        """Load and preprocess the Boston housing dataset."""
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = DataLoader.get_data(data_url)
        X = np.hstack([raw_df.values[::2, :-1], raw_df.values[1::2, :2]])
        y = raw_df.values[1::2, 2].reshape(-1, 1).ravel()
        return X, y

    @staticmethod
    def load_california_data():
        from sklearn.datasets import fetch_california_housing

        housing = fetch_california_housing()
        return housing.data, housing.target
