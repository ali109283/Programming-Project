# Just the run file of the code
# We start by importing our Classes & Functions

from src.Class_Data import Data 
from src.Class_Panel import Panel
from src.year_conv import years
from src.kpi import kpi_table
from src.ols_regression import regression
from src.adf import adf_test
from src.Class_LSTM import LuxuryLSTM

start = years(N=3) # Select number of years for the data training

# First we run the parent Class Data for one ticker mostly for a sanity check but also to show some first data and how it works standalone

LVMH = Data("MC.PA" ,start)
LVMH.process()
LVMH.get_data()
LVMH.get_kpis()
LVMH.plot()

# Here we define the luxury sector by selecting the top maisons, this is of course tunable

luxury_tickers = {

    "LVMH": "MC.PA",
    "Hermès": "RMS.PA",
    "Kering": "KER.PA",
    "Richemont": "CFR.SW",
    "Swatch": "UHR.SW",
    "Burberry": "BRBY.L",
    "Prada": "1913.HK"
}

# Then we generate the Panel data from the Class Panel (Child of Class Data) and show first resuslts of the sector
# Apart from the data we have four plots, consolidated logreturns and rolling volatility as well as the average of those sector wise

panel = Panel(luxury_tickers, years=3)
panel.build()
panel.plot_panel()
panel.get_panel()
panel.compute_sector_averages()
panel.plot_sector_averages()

# Then we get some KPIs for each maison 

#kpis = kpi_table(luxury_tickers) # This will take some time and show some threads let it run

# Afterwards we run an OLS regression to check the correlation between company and sector logarithmic returns and volatility

company = LVMH.get_data() # Here I use again LVMH, tunable 
sector = panel.compute_sector_averages() # Just to have it in a new data set not don't touch the original one
regression(company, sector,"LVMH")

# The ADF test

adf = panel.get_panel()
adf_test(adf)

# Lastly we initiate the Class LSTM

nas = LuxuryLSTM(panel_data= panel.get_panel(), sector_features= sector)
nas.run_NAS(n_architectures=1) # Here we chose how many architectures we want to create and assess the best one based on MSE
nas.retrain_best_models_all() # Retrain all the companies based the best achitecture

# Live signals
live_df = nas.live_signals_all(signal_thresh=0.005, ret_clip=3)

# Backtest last 2 weeks
backtest_df = nas.backtest_recent(signal_thresh=0.005, lookback_days=5)
summary = nas.backtest_summary(backtest_df)


