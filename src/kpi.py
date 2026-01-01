# Small code that helps generate the KPIs

import pandas as pd
from tabulate import tabulate
from src.Class_Data import Data # Ikmport Data Class to generate the KPIs


from tabulate import tabulate

luxury_tickers = {
    "LVMH": "MC.PA",
    "Hermès": "RMS.PA",
    "Kering": "KER.PA",
    "Richemont": "CFR.SW",
    "Swatch": "UHR.SW",
    "Prada": "1913.HK"
}


def kpi_table(luxury_tickers: dict, years: int = 5):

    """
    Generates a clean KPI table for a set of tickers using the Data class.

    """

    # Storage for KPI results
    kpi_results = {}

    for name, ticker in luxury_tickers.items():
        try:
            stock = Data(ticker, years)
            stock.process()
            kpis = stock.get_kpis()

            # Replace None values
            kpis_clean = {k: (v if v is not None else "NA") for k, v in kpis.items()}
            kpi_results[name] = kpis_clean

        except Exception as e:
            print(f" Failed for {name}: {e}")
            # fallback values
            kpi_results[name] = {
                "Dividend_Yield": "NA",
                "Book_Value": "NA",
                "Market_to_Book": "NA",
                "Profit_Margin": "NA",
                "Debt_to_Equity": "NA"
            }

    # Convert dictionary to DataFrame
    df_kpis = pd.DataFrame(kpi_results)
    df_kpis.index.name = "KPI"

    # Reorder companies if available
    desired_order = ["LVMH", "Hermès", "Kering", "Richemont", "Swatch", "Prada"]
    existing = [c for c in desired_order if c in df_kpis.columns]
    df_kpis = df_kpis[existing]

    # Print formatted table
    title = "Luxury Sector KPI Panel"
    print("\n" + title.center(80, " ") + "\n")
    print(tabulate(df_kpis, headers='keys', tablefmt='grid', showindex=True))

    return df_kpis
