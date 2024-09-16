from twsq.alpha import Alpha
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CTB(Alpha):
    def get_high(self, symbol, n):
        bars = self.get_lastn_bars(symbol + '/USD', n, '1d')
        return bars['high']
    
    def get_low(self, symbol, n):
        bars = self.get_lastn_bars(symbol + '/USD', n, '1d')
        return bars['low']
    
    def prepare(self, symbols, **kwargs):
        self.symbols = symbols
        self.entry_lookback = kwargs.get('entry_lookback')
        self.exit_lookback = kwargs.get('exit_lookback')
        
    def rebalance(self):
        for symbol in self.symbols:
            high = self.get_high(symbol, self.entry_lookback)
            low = self.get_low(symbol, self.exit_lookback)
            current_high = high.iloc[-1]
            current_low = low.iloc[-1]
            highest_high = high.max()
            lowest_low = low.min()
            
            # check if we have a position
            pos = self.get_pos()
            symbol_pos = pos.get(symbol, 0)
            
            if symbol_pos == 0:
            # if current high >= past Y days highest high, then buy
                if current_high >= highest_high:
                    symbol_qty = 1000 / self.get_current_price(symbol + '/USD')
                    self.create_order(symbol + '/USD', symbol_qty, 'buy', route=True)
            else:
            # Once bought, if current low <= past X days lowest low, then sell
                if current_low <= lowest_low:
                    self.create_order(symbol + '/USD', abs(symbol_pos), 'sell', route=True)
    def on_exit(self):
        self.cancel_all_orders() 
            
# Top 10 symbols to test
symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']

result = CTB.run_backtest(start_ts='20230701', freq='1d',  symbols=symbols, entry_lookback=28, exit_lookback=7)
 
# rebalance_freq = '1d'
# result = CTB.run_backtest(start_ts='20230701', freq=rebalance_freq,  symbols=symbols, entry_lookback=28, exit_lookback=7)

def compute_stats(results):
    pnl = results['pnl']
    stats = {}
    stats['sharpe'] = np.sqrt(252) * np.mean(pnl) / np.std(pnl)
    stats['avg_pnl'] = np.mean(pnl)
    stats['volatility'] = np.std(pnl) * np.sqrt(252)
    stats['max_drawdown'] = drawdown(pnl).min()
    stats['max_drawdown_duration'] = drawdown_duration(pnl).max()
    stats['total_pnl'] = pnl.sum()
    return stats
    
    
def drawdown(returns):
    cum_rets = returns.cumsum()
    hwm = cum_rets.cummax()
    drawdowns = cum_rets - hwm
    return drawdowns

def drawdown_duration(pnl):
    cum_pnl = pnl.cumsum()
    hwm = cum_pnl.cummax()
    
    duration = np.zeros(len(pnl))
    for t in range(1, len(pnl)):
        if cum_pnl.iloc[t] < hwm.iloc[t]:  # If we are in a drawdown
            duration[t] = duration[t-1] + 1  # Increment the duration
        else:
            duration[t] = 0  # Reset duration if recovered to HWM

    return pd.Series(duration, index=pnl.index)

stats = compute_stats(result.pos_pnl)
stats_df = pd.DataFrame(stats, index=[0])
print(stats_df)

plt.plot(result.pos_pnl['pnl'].cumsum())
plt.show()

stats_df.to_csv('ctb_stats.csv')

result = CTB.run_backtest(start_ts='20230701', freq='1d',  symbols=symbols, entry_lookback=28, exit_lookback=7)  
CTB.run_live(freq='1d', symbols=symbols, entry_lookback=28, exit_lookback=7)