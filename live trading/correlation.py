from twsq.alpha import Alpha
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Correlation(Alpha):
    
    def get_correlation(self, symbol1, symbol2): 
        # Get the last 2 years of daily bars for both symbols
        bars1 = self.get_lastn_bars(symbol1 + '/USD', 365*2, '1d')
        bars2 = self.get_lastn_bars(symbol2 + '/USD', 365*2, '1d')
        return bars1['close'].corr(bars2['close'])
    
    def select_top_pairs(self, symbols, n=3):
        pairs = []
        seen_pairs = set()  # To keep track of seen pairs and avoid duplicates
        for i in range(len(symbols)):
            symbol1 = symbols[i]
            corr_list = []
            for j in range(len(symbols)):
                if i != j:
                    symbol2 = symbols[j]
                    # Ensure pairs are unique and not duplicated
                    if (symbol1, symbol2) not in seen_pairs and (symbol2, symbol1) not in seen_pairs:
                        corr = self.get_correlation(symbol1, symbol2)
                        if abs(corr) >= 0.9:
                            corr_list.append((symbol2, abs(corr)))
                            seen_pairs.add((symbol1, symbol2))
            corr_list.sort(key=lambda x: x[1], reverse=True)
            top_n = corr_list[:n]
            for symbol, corr in top_n:
                pairs.append((symbol1, symbol, corr))
                print(f'Selected pair: ({symbol1}, {symbol}) with correlation: {corr}')
        return pairs
    
    def calculate_z_score(self, symbol1, symbol2):
        # Get the last 90 days of daily bars for both symbols
        bars1 = self.get_lastn_bars(symbol1 + '/USD', 90, '1d')
        bars2 = self.get_lastn_bars(symbol2 + '/USD', 90, '1d')
        
        # Ensure both bars have the same length
        min_length = min(len(bars1), len(bars2))
        if min_length < 90:
            raise ValueError(f"Not enough data to calculate Z-score for pair: {symbol1}, {symbol2}")
        
        # Align the lengths of both series
        log_prices1 = np.log(bars1['close'][-min_length:])
        log_prices2 = np.log(bars2['close'][-min_length:])
        
        # calculate the covaiance, beta and alpha
        cov = np.cov(log_prices1, log_prices2)[0][1]
        beta = cov / np.var(log_prices2)
        alpha = np.mean(log_prices1) - beta * np.mean(log_prices2)
        # calculate the spread
        spread = log_prices1 - beta * log_prices2 - alpha
        # calulate the mean and std of spread
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)
        # calculate the z-score
        z_score = (spread.iloc[-1] - mean_spread) / std_spread
        return z_score, beta
    
    def prepare(self, symbols, **kwargs):
        self.symbols = kwargs.get('symbols', symbols)
        self.top_pairs = self.select_top_pairs(self.symbols, n=kwargs.get('top_n', 3))
        if not self.top_pairs:
            raise ValueError('No pairs found with correlation >= 0.9')
        print(f'Number of pairs selected: {len(self.top_pairs)}')
        self.entry_threshold = kwargs.get('entry_z', 1)
        self.exit_threshold = kwargs.get('exit_z', 0.1)
        self.positions = {}
        for symbol1, symbol2, _ in self.top_pairs:
            self.positions[(symbol1, symbol2)] = (0, 0)
        
    def rebalance(self):
        for symbol1, symbol2, _ in self.top_pairs:
            try:        
                z_score, beta = self.calculate_z_score(symbol1, symbol2)
                current_position = self.positions[(symbol1, symbol2)]
                if beta < 0:
                    beta = abs(beta)
                    if current_position == (0, 0) and z_score > self.entry_threshold:
                        # Create orders
                        print(f"Opening position: SELL {symbol1}, SELL {symbol2} at Z-score: {z_score:.2f}")
                        self.create_order(symbol1 + '/USD', 1, 'sell', route=True)
                        self.create_order(symbol2 + '/USD', beta, 'sell', route=True)
                        self.positions[(symbol1, symbol2)] = (-1, -beta)
                    elif current_position == (0, 0) and z_score < -self.entry_threshold:
                        print(f"Opening position: BUY {symbol1}, BUY {symbol2} at Z-score: {z_score:.2f}")
                        self.create_order(symbol1 + '/USD', 1, 'buy', route=True)
                        self.create_order(symbol2 + '/USD', beta, 'buy', route=True)
                        self.positions[(symbol1, symbol2)] = (1, beta)
                else:
                    if current_position == (0, 0) and z_score > self.entry_threshold:
                        print(f"Opening position: SELL {symbol1}, BUY {symbol2} at Z-score: {z_score:.2f}")
                        self.create_order(symbol1 + '/USD', 1, 'sell', route=True)
                        self.create_order(symbol2 + '/USD', beta, 'buy', route=True)
                        self.positions[(symbol1, symbol2)] = (-1, beta)
                    elif current_position == (0, 0) and z_score < -self.entry_threshold:
                        print(f"Opening position: BUY {symbol1}, SELL {symbol2} at Z-score: {z_score:.2f}")
                        self.create_order(symbol1 + '/USD', 1, 'buy', route=True)
                        self.create_order(symbol2 + '/USD', beta, 'sell', route=True)
                        self.positions[(symbol1, symbol2)] = (1, -beta)
                # Close the pair position if the z-score reverts back to the exit threshold
                if current_position != (0, 0) and abs(z_score) < self.exit_threshold:
                    pos = self.get_pos()
                    symbol1_pos = pos.get(symbol1, 0)
                    symbol2_pos = pos.get(symbol2, 0)
                    print(f'Closing pair ({symbol1}, {symbol2}) with positions: {symbol1_pos}, {symbol2_pos} at Z-score: {z_score:.2f}')
                    # Adjust the positions to close the trade
                    target = {
                        symbol1: symbol1_pos - current_position[0], 
                        symbol2: symbol2_pos - current_position[1]
                    }
                    self.trade_to_target(target, route=True)
                    self.positions[(symbol1, symbol2)] = (0, 0)
            except Exception as e:
                print(f'Error rebalancing pair ({symbol1}, {symbol2}): {e}')  
    
    def on_exit(self):
        self.cancel_all_orders() 

# Top 10 symbols to test
symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'LTC', 'LINK']

result = Correlation.run_backtest(start_ts='20240101', freq='1d',
                                  symbols=symbols, top_n=3, entry_z=1, exit_z=0.1)

# performance evaluation
def compute_stats(results):
    rets = results['port_val'] / results['port_val'].shift() - 1
    dd = drawdown(rets.cumsum().fillna(0))
    ddd = duration(rets.cumsum().fillna(0))
    stats={}
    stats['sharpe'] = rets.mean() / rets.std() * np.sqrt(252)
    stats['volatility'] = rets.std() * np.sqrt(252)
    stats['total_pnl'] = results['pnl'].sum()
    stats['max_drawdown'] = dd.min()
    stats['max_duration'] = ddd.max()
    return stats
   
def drawdown(px):
    return px / px.expanding(min_periods=1).max() - 1
    
def duration(px):
    peak = px.expanding(min_periods=1).max()
    res = pd.Series(0, index=px.index)
    for dt in px.index:
        if px.loc[dt] >= peak.loc[dt]:
            res.loc[dt] = 0
        else: 
            res.loc[dt] = res.loc[:dt].iloc[-2] + 1
    return res

print(result.pos_pnl)
stats = compute_stats(result.pos_pnl)
print(stats)

# plot the portfolio value
plt.plot(result.pos_pnl['port_val'])
plt.show()
    
# plot the daily returns
rets = result.pos_pnl['port_val'] / result.pos_pnl['port_val'].shift() - 1
plt.plot(rets.cumsum())
plt.show()

# plot max drawdown
dd = drawdown(rets.cumsum())
plt.plot(dd)
plt.show()

# Correlation.run_live(freq='1d', symbols=symbols, top_n=3, entry_z=1, exit_z=0.1)


rets = result.pos_pnl['port_val'] / result.pos_pnl['port_val'].shift() - 1

