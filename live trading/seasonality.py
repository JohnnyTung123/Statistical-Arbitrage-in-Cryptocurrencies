from twsq.alpha import Alpha
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Seasonality(Alpha):
    def rebalance(self):
        # Define the target cryptocurrency
        symbol = 'BTC/USD'
        
        # Get the timestamp of the current bar
        current_time = self.ts

        # Check if current time is 21:00 UTC+0 (buy time)
        if current_time.hour == 21 and current_time.minute == 0:
            # Buy Bitcoin at 21:00 UTC+0
            # print(f"Buying {symbol} at {current_time} UTC+0")
            self.create_order(symbol, 1, 'buy', route=True)  # Adjust the quantity as needed

        # Check if current time is 22:00 UTC+0 (sell time)
        elif current_time.hour == 22 and current_time.minute == 0:
            # Sell Bitcoin at 22:00 UTC+0
            # print(f"Selling {symbol} at {current_time} UTC+0")
            self.create_order(symbol, 1, 'sell', route=True)  # Adjust the quantity as needed

    def on_exit(self):
        # Cancel all orders when exiting the strategy
        self.cancel_all_orders()


# gross = Seasonality.run_backtest(start_ts='20200101', end_ts='20211231', name = 'Gross',freq='1h', 
# 	taker_fee = 0, maker_fee = 0, slip = 0)
# net = Seasonality.run_backtest(start_ts='20200101', end_ts='20211231', name = 'Net',freq='1h')

result = Seasonality.run_backtest(start_ts='20230101', freq='1h')