# In this part we create another Class name Panel child of Data 
# which will beused as a facilitarot to run smoothly Class Data 
# using all the seven tickers requred for the panel


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.Class_Data import Data
from src.year_conv import years

# In this part we create another Class name Panel child of Data 
# which will beused as a facilitarot to run smoothly Class Data 
# using all the seven tickers requred for the panel


class Panel:

    """
    A class to that facilitates data generation for multiple
    using the parent Data class as a building block.
    """

    def __init__(self, tickers_dict: dict, years = 5):

        """
        Initialize the PanelData object.
        """
        self.tickers_dict = tickers_dict
        self.years_back = years
        self.data_objects = {}
        self.panel = None


    def build(self):

        """
        Fetch and process data for each company in the panel.
        """
        start_date = years(self.years_back)

        combined = {}

        for name, ticker in self.tickers_dict.items():

            obj = Data(ticker, start=start_date)
            obj.process()
            df = obj.get_data()
            #self.data_objects[name] = obj
        

        # Combine into one multi-index DataFrame

            combined[name] = df[["Close", "LogReturn", "Volatility"]]

        self.panel = pd.concat(combined, axis=1)

        # Its makes more sense to have the latest data first

        self.panel = self.panel.sort_index(ascending=False)


    def add_data_object(self, name: str, data_obj):
        
        """
        Add a Data object to the panel.
        """
        self.data_objects[name] = data_obj

    
    def plot_panel(self):

        """
        Plot comparative log returns and volatilities for all firms.
        """

        plt.figure(figsize=(14, 10))
        plt.suptitle("Luxury Sector Stock Analysis", fontsize=16, fontweight="bold")

        # 1st Plot. Volatility Comparison

        plt.subplot(2, 1, 1)

        for name in self.tickers_dict.keys():
            plt.plot(
                self.panel[name]["Volatility"],
                label=name,
                linewidth=1.2
            )

        plt.title("Rolling Volatility (Annualized)")
        plt.ylabel("Volatility")
        plt.grid(True)
        plt.legend()

        # 2nd Plot. Log Return Comparison

        plt.subplot(2, 1, 2)

        for name in self.tickers_dict.keys():

            plt.plot(
                self.panel[name]["LogReturn"],
                label=name,
                linewidth=0.8
            )
        plt.title("Daily Log Returns")
        plt.ylabel("Log Return")
        plt.grid(True)
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


    # Store everything in a Pandas DataFrame

    def get_panel(self) -> pd.DataFrame:

        """
        Return the combined panel DataFrame and save raw data to CSV
        """

        self.panel.to_csv("Data/combined_panel.csv", index=True)
    
        return self.panel.copy()
    
    # Generate average of the seven companies

    def compute_sector_averages(self):
        """
        Compute the average volatility and log returns across the data set

        """

        # Extract relevant slices
        vol_df = self.panel.xs("Volatility", level=1, axis=1)
        ret_df = self.panel.xs("LogReturn", level=1, axis=1)

        # Compute averages across companies (row-wise mean)
        avg_vol = vol_df.mean(axis=1)
        avg_ret = ret_df.mean(axis=1)

        # Combine into single DataFrame
        avg_df = pd.DataFrame({
            "Avg_LogReturn": avg_ret,
            "Avg_Volatility": avg_vol
        })

        return avg_df

    def plot_sector_averages(self):

        """
        Plot the average log returns and volatility of the sector

        """
        avg_df = self.compute_sector_averages()

        # 1st Average Sector Volatility

        plt.figure()
        plt.plot(avg_df["Avg_Volatility"], color="blue", linewidth=1.3)
        plt.title(" Rolling Volatility Luxury Sector ", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        ax = plt.gca() # using mdate to show only one tick per year
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.ylabel("Volatility")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("Results/Average Volatility Luxury Sector.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 2nd Average Log Returns
        
        plt.figure()
        plt.plot(avg_df["Avg_LogReturn"], color="black", linewidth=1.3)
        plt.title("Daily Log Returns - Luxury Sector ", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        ax = plt.gca() # using mdate to show only one tick per year
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.ylabel("Log Return")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("Results/Average LogReurns Luxury Sector.png", dpi=300, bbox_inches="tight")
        plt.show()