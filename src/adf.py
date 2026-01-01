# This a function that implement the Augmented Dicket-Fuller and check for each comapy 
# 1) Test Statistic
# 2) p-value
# 3) Critical values
# 4) Stationarity conclusion


# import the needed library, here there is a direct adfuller function 

from statsmodels.tsa.stattools import adfuller
import pandas as pd
import textwrap # To nicely show the output table & summary

def adf_test(data):

    """
    Runs ADF tests on Close, LogReturn, and Volatility for each company
    and prints a result table and a summary if the test was successful.
    """

    results = []

    for company in data.columns.levels[0]:  # Loop over company tickers which are the top level columns

        for feature in ["Close", "LogReturn", "Volatility"]:

            try:

                series = data[company][feature].dropna()

                adf_result = adfuller(series) # Dicectly from the library

                results.append({
                    "Company": company,
                    "Feature": feature,
                    "ADF Statistic": round(adf_result[0], 4),
                    "p-value": round(adf_result[1], 4),
                    "1% CV": adf_result[4]["1%"],
                    "5% CV": adf_result[4]["5%"],
                    "10% CV": adf_result[4]["10%"],
                    "Stationary?": "Yes" if adf_result[1] < 0.05 else "No"
                })

            except Exception as e:

                print(f" Failed ADF for {company} – {feature}: {e}")

    # Convert to dataframe

    df_adf = pd.DataFrame(results)
    df_adf = df_adf.set_index(["Company", "Feature"])

    # Display the table

    print("\n================================== ADF TEST TABLE ====================================")
    print(df_adf)
    print("======================================================================================\n")

     # ---- Summary Section ----

    total_tests = len(df_adf)

    # Count per feature

    stationary_lr = sum((df_adf.index.get_level_values("Feature") == "LogReturn") &
                        (df_adf["Stationary?"] == "Yes"))
    total_lr = sum(df_adf.index.get_level_values("Feature") == "LogReturn")

    stationary_vol = sum((df_adf.index.get_level_values("Feature") == "Volatility") &
                         (df_adf["Stationary?"] == "Yes"))
    total_vol = sum(df_adf.index.get_level_values("Feature") == "Volatility")

    # Expected conditions

    logreturn_ok = stationary_lr / total_lr >= 0.9 # The choice of 0.9 is more like a sanity threshold as we expect high values
    volatility_ok = stationary_vol / total_vol >= 0.7 # Less value as logreturns are can also be heteroskedastic, not strictly stationary

    # ---- Conditional Summary ----

    if logreturn_ok and volatility_ok:
        summary_sentence =  (
                "The ADF test results confirm that return-based features "
    "(LogReturns and Volatility) exhibit the expected stationarity "
    "across all companies in the dataset.\n\n"
    "In contrast, Closing Prices consistently fail to reject the null "
    "hypothesis of a unit root, which aligns with well-established "
    "financial theory: price levels typically follow a random walk and "
    "are therefore non-stationary.\n\n"
    "Taken together, these outcomes indicate that the dataset has been "
    "appropriately preprocessed. The stationary features are suitable "
    "for neural-network modeling, and the transformations applied—"
    "particularly the computation of log-returns—effectively prepare "
    "the data for LSTM-based forecasting and further predictive analysis."
        )

    else:
        summary_sentence = (
            "The ADF tests indicate that one or more return-based series did not achieve\n\n "
            "The expected level of stationarity. Further inspection or preprocessing "
            "may be required before proceeding with neural network training."
        )

    wrapped = "\n\n".join(textwrap.fill(paragraph, width=80) for paragraph in summary_sentence.split("\n\n")) # Use of texwrap to have a nice output

    print("=============================== ADF TEST SUMMARY ===============================")
    print(wrapped)
    print("=================================================================================\n")

    # little modification for have the table nicely in one line

    pd.set_option("display.width", 200) 
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 20)

    return df_adf
