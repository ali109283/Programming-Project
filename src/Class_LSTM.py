# Here we create the Luxury LSTM Class

import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class LuxuryLSTM:

    """
    A class that essentially creates arbitrary  a number of NN architectures, validates them 
    per company and returns the one with the best MSE. Then all companies are trained with the
    selected architecture and prediction is made the next-day log-price, using both firm-level
    features and lagged sector indicators which are derived direcly from the panel data. For
    infernce, based on the logreturn output a method produces signals "Buy-Hold-Sell"applying 
    volatility-based clipping, and thresholding and signal-generation. Hence, live investment
    reccomendations are available. Lastly, we have the backtest method where we actual check for
    the last N days if the signals are followed what would have been our their accuracy and returns.
    """
    
    def __init__(self, panel_data, sector_features, window=30, verbose=True):
        
        self.panel = panel_data
        self.sector = sector_features
        self.window = window
        self.verbose = verbose

        self.best_arch = None
        self.models = {}       # Trained model
        self.scalers_X = {}    # X scalers
        self.scalers_y = {}    # Y scalers

    
    def generate_random_architecture(self):

        """
        Method fron which we can get the possible LSTM architectures - Randomly
        
        """
        return {
            "layers": random.choice([1, 2, 3]),
            "units": random.choice([32, 48, 64]),
            "dropout": random.choice([0.0, 0.1, 0.2]),
            "lr": random.choice([0.0005, 0.001]),
            "epochs": random.choice([10, 15, 20]),
            "batch": random.choice([16, 32])
        }

    
    def build_model(self, arch, input_dim):

        """
        Method to build the architectures.
        
        """

        model = Sequential()
        for i in range(arch["layers"]):

            if i == 0:
                model.add(LSTM(
                    arch["units"],
                    return_sequences=(i < arch["layers"] - 1),
                    input_shape=(self.window, input_dim)
                ))

            else:

                model.add(LSTM(
                    arch["units"],
                    return_sequences=(i < arch["layers"] - 1)
                ))
            if arch["dropout"] > 0:

                model.add(Dropout(arch["dropout"]))
                
        model.add(Dense(1))
        model.compile(loss="mse", optimizer=Adam(learning_rate=arch["lr"]))
        return model
    

    # Create rolling windows
    
    def create_xy(self, X, y):

        Xs, ys = [], []

        for i in range(len(X) - self.window):

            Xs.append(X[i:i+self.window])
            ys.append(y[i+self.window])

        return np.array(Xs), np.array(ys)
    

    # Train & Validation Split

    def split_xy(self, X, y, ratio=0.15):
        cut = int(len(X) * (1 - ratio))
        return X[:cut], X[cut:], y[:cut], y[cut:]

    # Evaluate MSE in one company seperately

    def evaluate_on_company(self, model, X_train, X_val, y_train, y_val, arch):

        model.fit(X_train, y_train, epochs=arch["epochs"], batch_size=arch["batch"], verbose=0)
        preds = model.predict(X_val, verbose=0)

        return mean_squared_error(y_val, preds)

    # Main loop that iterates to find the best arhcitecture.

    def run_NAS(self, n_architectures=5):
        
        results = []

        for i in range(n_architectures):

            arch = self.generate_random_architecture() # using the method
            mse_list = []

            for company in self.panel.columns.levels[0]:
                df = self.panel[company][["Close", "LogReturn", "Volatility"]].copy()
                df["Sector_Log"] = self.sector["Avg_LogReturn"].shift(1)
                df["Sector_Vol"] = self.sector["Avg_Volatility"].shift(1)
                df = df.sort_index() # just to be sure with the dates
                df["Log_Close"] = np.log(df["Close"])
                df["Target"] = df["Log_Close"].shift(-1)
                df = df.dropna()

                X_raw = df[["Log_Close", "LogReturn", "Volatility", "Sector_Log", "Sector_Vol"]].values
                y_raw = df["Target"].values.reshape(-1, 1)

                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X = scaler_X.fit_transform(X_raw)
                y = scaler_y.fit_transform(y_raw)

                X_win, y_win = self.create_xy(X, y)
                X_tr, X_val, y_tr, y_val = self.split_xy(X_win, y_win)

                model = self.build_model(arch, X_win.shape[2])
                mse = self.evaluate_on_company(model, X_tr, X_val, y_tr, y_val, arch)
                mse_list.append(mse)

            avg_mse = np.mean(mse_list)
            results.append((arch, avg_mse))

            if self.verbose:
                print(f"Testing architecture {i+1}/{n_architectures}: {arch}")
                print(f"   → Avg MSE: {avg_mse:.6f}")

        self.best_arch, best_mse = min(results, key=lambda x: x[1])
        print("BEST ARCHITECTURE SELECTED")
        print(self.best_arch)
        print(f"Average MSE: {best_mse:.6f}")
        return self.best_arch

    # Retrain best model for the companies

    def retrain_best_models_all(self):

        for company in self.panel.columns.levels[0]:

            df = self.panel[company][["Close", "LogReturn", "Volatility"]].copy()
            df["Sector_Log"] = self.sector["Avg_LogReturn"].shift(1)
            df["Sector_Vol"] = self.sector["Avg_Volatility"].shift(1)
            df = df.sort_index()
            df["Log_Close"] = np.log(df["Close"])
            df["Target"] = df["Log_Close"].shift(-1)
            df = df.dropna()

            X_raw = df[["Log_Close", "LogReturn", "Volatility", "Sector_Log", "Sector_Vol"]].values
            y_raw = df["Target"].values.reshape(-1,1)

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X = scaler_X.fit_transform(X_raw)
            y = scaler_y.fit_transform(y_raw)

            X_win, y_win = self.create_xy(X, y)

            model = self.build_model(self.best_arch, X_win.shape[2])
            model.fit(X_win, y_win, epochs=self.best_arch["epochs"], batch_size=self.best_arch["batch"], verbose=0)

            self.models[company] = model
            self.scalers_X[company] = scaler_X
            self.scalers_y[company] = scaler_y

    # Generate the singals for the next trading day

    def live_signals_all(self, signal_thresh=0.005, ret_clip=3):

        rows = []

        for company in self.panel.columns.levels[0]:

            if company not in self.models:
                continue

            df = self.panel[company][["Close", "LogReturn", "Volatility"]].copy()
            df["Sector_Log"] = self.sector["Avg_LogReturn"].shift(1)
            df["Sector_Vol"] = self.sector["Avg_Volatility"].shift(1)
            df = df.sort_index()
            df["Log_Close"] = np.log(df["Close"])
            df = df.dropna()

            if len(df) < self.window:
                continue

            window_raw = df[["Log_Close", "LogReturn", "Volatility", "Sector_Log", "Sector_Vol"]].values[-self.window:].copy()
            window_scaled = self.scalers_X[company].transform(window_raw)

            last_log_price = df["Log_Close"].iloc[-1]
            next_date = df.index[-1] + pd.tseries.offsets.BDay(1)

            pred_scaled = self.models[company].predict(window_scaled.reshape(1, self.window, -1), verbose=0)[0, 0]
            pred_log = self.scalers_y[company].inverse_transform([[pred_scaled]])[0, 0]

            ret = pred_log - last_log_price
            hist_vol = df["LogReturn"].iloc[-self.window:].std()
            ret = np.clip(ret, -ret_clip*hist_vol, ret_clip*hist_vol)

            signal = "BUY" if ret > signal_thresh else "SELL" if ret < -signal_thresh else "HOLD"

            rows.append([next_date, company, ret, signal])

        df = pd.DataFrame(rows, columns=["Date","Ticker","Pred_Return","Signal"]).sort_values(["Ticker"])
        print(df)

        return df
    

    # Backtest method, for N amount of days - fixed for 5 to keep it short to avoid drifting

    def backtest_recent(self, signal_thresh=0.005, lookback_days=5):
        
        rows = []

        for company in self.panel.columns.levels[0]:

            df = self.panel[company][["Close", "LogReturn", "Volatility"]].copy()
            df["Sector_Log"] = self.sector["Avg_LogReturn"].shift(1)
            df["Sector_Vol"] = self.sector["Avg_Volatility"].shift(1)
            df = df.sort_index()
            df["Log_Close"] = np.log(df["Close"])
            df = df.dropna()

            if len(df) < self.window + lookback_days:
                continue

            # Look back last N days

            for i in range(-lookback_days, 0):

                window_raw = df[["Log_Close", "LogReturn", "Volatility", "Sector_Log", "Sector_Vol"]].values[i-self.window:i].copy()
                window_scaled = self.scalers_X[company].transform(window_raw)

                last_log_price = df["Log_Close"].iloc[i-1]
                pred_scaled = self.models[company].predict(window_scaled.reshape(1, self.window, -1), verbose=0)[0, 0]
                pred_log = self.scalers_y[company].inverse_transform([[pred_scaled]])[0, 0]

                ret = pred_log - last_log_price
                signal = "BUY" if ret > signal_thresh else "SELL" if ret < -signal_thresh else "HOLD"

                actual_ret = df["Log_Close"].iloc[i] - last_log_price

                rows.append([df.index[i], company, ret, signal, actual_ret])

        df = (pd.DataFrame(rows,columns=["Date", "Ticker", "Pred_Return", "Signal", "Actual_Return"]).sort_values(["Ticker", "Date"]))
        print(df)

        return df

    # Little Method to have a clear summary table


    def backtest_summary(self, backtest_df, hold_thresh=0.002):

        df = backtest_df.copy()

        def hit_signal(row):

            if row['Signal'] == 'BUY':
                return row['Actual_Return'] > 0
            
            elif row['Signal'] == 'SELL':
                return row['Actual_Return'] < 0
            
            else:  # HOLD
                return abs(row['Actual_Return']) <= hold_thresh

        df['Hit'] = df.apply(hit_signal, axis=1)

        def pnl_signal(row):

            if row['Signal'] == 'BUY':
                return row['Actual_Return']
            elif row['Signal'] == 'SELL':
                return -row['Actual_Return']
            else:
                return 0

        df['PnL'] = df.apply(pnl_signal, axis=1)

        signal_accuracy = df['Hit'].mean()
        avg_pnl = df['PnL'].mean()
        cumulative_pnl = df['PnL'].sum()

        per_ticker = df.groupby('Ticker')['PnL'].agg(['mean','sum']).reset_index()
        per_ticker.rename(columns={'mean':'Avg_PnL','sum':'Cumulative_PnL'}, inplace=True)

        # Was a bit messy to find the correct format
        print("")
        label_width = 25
        print("Backtest Summary")
        print("="*33)
        print(f"{'Signal Accuracy':<{label_width}}: {signal_accuracy:.2%}")
        print(f"{'Average PnL':<{label_width}}: {avg_pnl*100:+.2f}%")
        print(f"{'Cumulative PnL':<{label_width}}: {cumulative_pnl*100:+.2f}%\n")


        # Per Maison Perfomance
        print("Per Maison Performance:")
        print("-"*33)
        formatted_table = per_ticker.copy()
        formatted_table['Avg_PnL'] = (formatted_table['Avg_PnL']*100).map("{:+.2f}%".format)
        formatted_table['Cumulative_PnL'] = (formatted_table['Cumulative_PnL']*100).map("{:+.2f}%".format)
        print(formatted_table.to_string(index=False))
        print("="*33 + "\n")

        return {
            'signal_accuracy': signal_accuracy,
            'avg_pnl': avg_pnl,
            'cumulative_pnl': cumulative_pnl,
            'per_ticker': per_ticker,
            'full_df': df
        }
