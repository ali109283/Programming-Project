# In this part we will create a Class Data to retreive multiple data.
# We will use data of the past 5 years - of course this can be easily modified

# Here we will need 4 libraries:

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.year_conv import years
import os


# Creation of Class Data

class Data:

    """
      Class to hadnle smooth data extraction 
    and preprocessing
    """

    def __init__(self, ticker, start, end=None):

        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = None # This will hold the downloaded data
        self.info = None # This is for the KPIs

    def yahoo(self):

        """
        Fetch historical data based on the given ticker
        """

        self.data = yf.download(self.ticker, start = self.start, end=self.end)

        # Kepp only the relevant columns

        self.data = self.data[["Close", "Volume"]]

        # Add the KPIs
        ticker_obj = yf.Ticker(self.ticker)
        self.info = ticker_obj.info


    def log_returns(self):

        """
        Calculate Daily Log returns as:
        long_return_T = ln(Close_T / Close_T-1)
        """

        self.data["LogReturn"] =  np.log(self.data["Close"]/ self.data["Close"].shift(1))


    def volatility(self, window: int=20):

        """
        Compute Rolling volatility, I use the rolling size
        as 20 days, its also the default - Also as trading days 
        per year are 252, we plug in the formula
        """

        self.data["Volatility"] = self.data["LogReturn"].rolling(window=window).std() * np. sqrt(252)


    def process(self):

        """
        Method to run and process all the data
        """

        self.yahoo()
        self.log_returns()
        self.volatility()


    def get_data(self) -> pd.DataFrame:


        """
        Method to derive the data once the object is initiated
        The data is returned in a pandas dataFrama for easier handling
        Also saves the data frame as csv file to the indicated folder
        """

        self.data.to_csv("Data/Company Data.csv", index=True)
        print(self.data)

        return self.data

    def get_kpis(self) -> dict:

        """
        Extract some key KPIs for the company directly from Yahoo finnace

        """

        kpis = {
            "Dividend_Yield": round(self.info.get("dividendYield"),3),
            "Book_Value": round(self.info.get("bookValue"),3),
            "Market_to_Book": round(self.info.get("currentPrice") / self.info.get("bookValue"),3),
            "Profit_Margin": round(self.info.get("profitMargins"),3),
            "Debt_to_Equity": round(self.info.get("debtToEquity"),3),
        }


        return kpis
    

    # Table creation for the KPIs

    def kpi_table(self):

        rows = {name: stock.get_kpis() for name, stock in self.stocks.items()}
        df = pd.DataFrame(rows).T
        print(df)
        
        return df
    

    def plot(self):

        """
        Plot method to displayclosing price, log returns, 
        and rolling volatility.

        """

        # 1st Graph Closing Price for the past N years

        plt.figure()
        plt.title(f"Stock Price - {self.ticker}")
        plt.plot(self.data["Close"], label="Closing Price", linewidth=1)
        plt.ylabel("Price")
        plt.xlabel("Year")
        ax = plt.gca() # using mdate to show only one tick per year
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.savefig(f"Results/Stock Price - {self.ticker}.png")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        # 2nd Graph Log Returns
        plt.figure()
        plt.title(f"Log Returns - {self.ticker}")
        plt.plot(self.data["LogReturn"], label="Return", color="green", linewidth=1)
        plt.ylabel("Log Return")
        plt.xlabel("Year")
        ax = plt.gca() # using mdate to show only one tick per year
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.savefig(f"Results/Log Returns - {self.ticker}.png")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        # 3rd Rolling Volatility
        plt.figure()
        plt.title(f"Rolling Volatility - {self.ticker}")
        plt.plot(self.data["Volatility"], label="20D Rolling Volatility", color="purple", linewidth=1)
        plt.ylabel("Volatility")
        plt.xlabel("Year")
        ax = plt.gca() # using mdate to show only one tick per year
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.savefig(f"Results/Volatility - {self.ticker}.png")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        


############################## This is helper to avoid the Yahoo Finance API overload - limit request found a good old stackoverflow

#https://stackoverflow.com/questions/79692010/yfinance-rate-limit-error-when-using-date-variables-but-not-hardcoded-dates

from curl_cffi import requests 

session = requests.Session(impersonate="chrome")

ticker = yf.Ticker("MC.PA", session=session)

#stock_data = yf.download("MC.PA", session=session)