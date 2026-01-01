# README File

# Programming_project

# Luxury Stock Market


The process is then the following :
1) Class Data retrieves all relevant information from yahoo finance and computes LogReturn and Rolling Volatility
2) Class Panel used Class Data to consolidate everything for the panel and computes the sector averages
3) OLS regression to check the correlations
4) Run the ADF test to check stationarity
5) Class_LSTM trains the model in N different architectures and choses the one with best MSE
6) After training is done for all tickers Signals and Bactest Summary is produced


#### Possible improvements:
1) NAS
2) KPIs
3) Different Strategies

## Code
* **main.py** contains the example of how to use the code, calling the classes and fucntions
* **Class_Data.py** contains the class from which we derive all the relevant data for one ticket
* **Class_Panel.py** contains the class that uses Class Data and creates the panel as well the sector averages
* **kpi.py** is the funtion that retrieves the KPIs for each ticker
* **ols_regression.py** contains a function that implements the OLS regression
* **adf.py** the ADF test function
* **year_conv.py** little function for dates handling
* **Class_LSTM.py** contains the class where we train the algorithm, chose the best architecture based on MSE, backtest, and produce signals

## Data

Data is directly retrieved dynamicly from yahoo finance library and stored and the Data folder

## Run order
1) *main.py*

That's it !

## Results
The plots can be found as .png files in the folder **Plots**.

**Warning : Based on the number of Architectures chose the algorithm can takle from min 5-7 to up to 30 mins to run**
