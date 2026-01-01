# This a relatively simple function where we run the OLS regression directly via the stats model

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt



def regression(company_df: pd.DataFrame, sector_df: pd.DataFrame, ticker):

    """
    Regressions of company log-returns and volatility against sector averages.

    """

    company_df.columns = company_df.columns.get_level_values(0) # Had a small issue with the aligment of the data frames


    # Just ensuring alignment for the two data sets

    df = pd.concat([
        company_df[['LogReturn', 'Volatility']],
        sector_df[['Avg_LogReturn', 'Avg_Volatility']]
    ], axis=1).dropna()

    # Log-returns regression

    X_ret = sm.add_constant(df['Avg_LogReturn'])
    y_ret = df['LogReturn']
    ret_model = sm.OLS(y_ret, X_ret).fit()

    # Volatility regression

    X_vol = sm.add_constant(df['Avg_Volatility'])
    y_vol = df['Volatility']
    vol_model = sm.OLS(y_vol, X_vol).fit()

    # Print results

    print(f"\n{ticker} vs Sector Regression Results")
    print("====================================\n")

    print(f"Log-Returns Regression {ticker}")
    print(f"Alpha (intercept) : {ret_model.params['const']:.4f}")
    print(f"Beta (sector)    : {ret_model.params['Avg_LogReturn']:.4f}")
    print(f"R²               : {ret_model.rsquared:.3f}")
    print(f"t-stat (beta)    : {ret_model.tvalues['Avg_LogReturn']:.2f}")
    print(f"p-value (beta)   : {ret_model.pvalues['Avg_LogReturn']:.4f}")
    print(f"Observations     : {int(ret_model.nobs)}\n")

    print(f"Volatility Regression {ticker}")
    print(f"Alpha (intercept) : {vol_model.params['const']:.4f}")
    print(f"Beta (sector)    : {vol_model.params['Avg_Volatility']:.4f}")
    print(f"R²               : {vol_model.rsquared:.3f}")
    print(f"t-stat (beta)    : {vol_model.tvalues['Avg_Volatility']:.2f}")
    print(f"p-value (beta)   : {vol_model.pvalues['Avg_Volatility']:.4f}")
    print(f"Observations     : {int(vol_model.nobs)}")

    # Plot: Log-returns

    plt.figure()
    plt.scatter(df['Avg_LogReturn'], y_ret, alpha=0.5)
    plt.plot(df['Avg_LogReturn'],ret_model.predict(X_ret),linewidth=2)
    plt.title(f"{ticker} vs Sector — Log-Returns")
    plt.xlabel("Sector Log-Return")
    plt.ylabel(f"{ticker} Log-Return")
    plt.savefig(f"Results/Log Returns Regression.png")
    plt.show()

    # Plot: Volatility
  
    plt.figure()
    plt.scatter(df['Avg_Volatility'], y_vol, alpha=0.5)
    plt.plot(df['Avg_Volatility'],vol_model.predict(X_vol),linewidth=2)
    plt.title(f"{ticker} vs Sector — Volatility")
    plt.xlabel("Sector Volatility")
    plt.ylabel(f"{ticker} Volatility")
    plt.savefig(f"Results/Volatility Regression.png")
    plt.show()

  
