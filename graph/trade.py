import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
from binance.client import Client


class Trade:
    # Binance apply a 0.1% fee for every trade
    # And a 25% discount if use BNB to pay the fee
    # BINANCE_OP_COST = 0.075
    BINANCE_OP_COST = 0.1
    def __init__(self):
        pass

    def get_symbols(self):
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()

        symbols_eur = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'EUR']
        return symbols_eur

    def get_candlesticks(self, symbol, days, interval="15m"):
        # API key e secret non necessarie per dati pubblici
        client = Client()

        # Parametri
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=days)

        # Recupera i dati delle candele (klines)
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_time.strftime("%d %b, %Y %H:%M:%S"),
            end_time.strftime("%d %b, %Y %H:%M:%S")
        )
        open_prices = [float(rec[1]) for rec in klines]
        close_prices = [float(rec[4]) for rec in klines]        

        return [[float(rec[1]),float(rec[4])] for rec in klines]

    # def calculate_rsi(data, period):
    #     deltas = np.diff(data)
    #     seed = deltas[:period]
    #     up = seed[seed >= 0].sum() / period
    #     down = -seed[seed < 0].sum() / period
    #     rs = up / down if down != 0 else 0
    #     rsi = np.zeros_like(data)
    #     rsi[:period] = 100 - 100 / (1 + rs)
    #     for i in range(period, len(data)):
    #         delta = deltas[i - 1]
    #         upval = max(delta, 0)
    #         downval = -min(delta, 0)
    #         up = (up * (period - 1) + upval) / period
    #         down = (down * (period - 1) + downval) / period
    #         rs = up / down if down != 0 else 0
    #         rsi[i] = 100 - 100 / (1 + rs)
    #     return rsi
    # def calculate_ema(data, period):
    #     alpha = 2 / (period + 1)
    #     ema = [np.mean(data[:period])]
    #     for price in data[period:]:
    #         ema.append(alpha * price + (1 - alpha) * ema[-1])
    #     return np.array(ema)

    def calculate_ema(self, candlesticks, period):
        closes = self.get_close_array_from_candlesticks(candlesticks)
        ema = []
        k = 2 / (period + 1)
        # Calcolo la SMA iniziale per i primi 'period' prezzi
        sma = sum(closes[:period]) / period
        ema.append(sma)

        # Calcolo EMA per i restanti prezzi
        for price in closes[period:]:
            ema_value = (price * k) + (ema[-1] * (1 - k))
            ema.append(ema_value)
        
        return ema

    # def calculate_macd(data, short_period, long_period, signal_period):
    #     ema_short = calculate_ema(data, short_period)
    #     ema_long = calculate_ema(data, long_period)

    #     min_len = min(len(ema_short), len(ema_long))
    #     macd_line = ema_short[-min_len:] - ema_long[-min_len:]
    #     signal_line = calculate_ema(macd_line, signal_period)

    #     macd_line = macd_line[-len(signal_line):]
    #     return macd_line, signal_line

    def avg_arr(self, arr):
        return sum(arr) / len(arr)

    def calculate_sma(self, candlesticks, period):
        closes = self.get_close_array_from_candlesticks(candlesticks)
        series = pd.Series(closes)
        sma = series.rolling(window=period).mean()
        return sma

    def calculate_smma(self, candlesticks, period):
        closes = pd.Series(self.get_close_array_from_candlesticks(candlesticks))
        smma = closes.rolling(window=period).mean()  # primo valore è la SMA iniziale
        for i in range(period, len(closes)):
            smma_i = (smma.iloc[i - 1] * (period - 1) + closes.iloc[i]) / period
            smma.iloc[i] = smma_i
        return smma

    def backtest_strategy(self,
                        candlesticks,
                        type,
                        period):

        if type == "sma":
            avg = self.calculate_sma(candlesticks, period)
        elif type == "smma":
            avg = self.calculate_smma(candlesticks, period)

        position = 0
        cash = 0
        entry_price = 0
        cost = 0
        ops = 0

        for i in range(period,len(candlesticks)):
            candlestick = candlesticks[i]
            open = candlestick[0]
            close = candlestick[1]
            if close < open:
                type_trade = "red"
            elif close > open:
                type_trade = "green"
            else:
                type_trade = "equal"
            if position == 0:
                if (type_trade == "green" and open > avg[i]):
                    print(i,type_trade,"BUY:",candlesticks[i-1],candlestick,avg[i], cash, cost)
                    position = 1
                    entry_price = close
                    cost += close * Trade.BINANCE_OP_COST / 100
                    ops += 1
                
            elif position == 1:
                if (type_trade == "red" and open < avg[i]):
                    print(i, type_trade,"SELL:",candlesticks[i-1],candlestick,avg[i], cash, cost)
                    cash += close - entry_price
                    position = 0
                    cost += close * Trade.BINANCE_OP_COST / 100
                    ops += 1

        if position == 1:
            cash += candlesticks[-1][1] - entry_price
            cost += close * Trade.BINANCE_OP_COST / 100
            ops += 1

        return cash, cost

    def optimize_params(self, data, avg):
        best_profit = -np.inf
        best_net = -np.inf
        best_period = None
        best_ops = None
        best_cost = None
        for period in range(5, 31, 5):
            profit, cost = self.backtest_strategy(data,
                                            avg,
                                            period)
            if profit - cost > best_net:
                best_net = profit - cost
                best_profit = profit
                best_period = period
                best_cost = cost

        return best_period, best_profit, best_cost

    def cross_ema_stragegy(self, candlesticks):
        ema_low = [0]*9 + self.calculate_ema(candlesticks, 9)
        ema_high = [0]*26 + self.calculate_ema(candlesticks, 26)

        position = 0
        cash = 0
        entry_price = 0
        cost = 0
        ops = 0

        for i in range(len(candlesticks)):
            candlestick = candlesticks[i]
            open = candlestick[0]
            close = candlestick[1]
            if close < open:
                type_trade = "red"
            elif close > open:
                type_trade = "green"
            else:
                type_trade = "equal"
            if position == 0:
                if type_trade == "green" and ema_low[i] > ema_high[i] and i > 26:
                    print(i,type_trade,"BUY:",candlestick, ema_low[i], ema_high[i], cash, cost)
                    position = 1
                    entry_price = close
                    cost += close * Trade.BINANCE_OP_COST / 100
                    ops += 1
                
            elif position == 1:
                if type_trade == "red" and ema_low[i] < ema_high[i] and i > 26:
                    print(i, type_trade,"SELL:",candlestick, ema_low[i], ema_high[i], cash, cost)
                    cash += close - entry_price
                    position = 0
                    cost += close * Trade.BINANCE_OP_COST / 100
                    ops += 1

        if position == 1:
            cash += candlesticks[-1][1] - entry_price
            cost += close * Trade.BINANCE_OP_COST / 100
            ops += 1

        return round(cash), round(ops), round(cost)

    def net_profit(self, best_profit, avg, days, best_ops):
        return f"{round(best_profit*100/avg/days-best_ops/days*Trade.BINANCE_OP_COST,1)}"    
        # return f"{round(best_profit*100/avg/days,1)}"    

    @staticmethod
    def get_close_array_from_candlesticks(candlesticks): 
        return [candlestick[1] for candlestick in candlesticks]

    @staticmethod
    def get_average_array_from_candlesticks(candlesticks): 
        return [np.mean(candlestick) for candlestick in candlesticks]

    def simulate(self):
        days = 30
        header = "N;Symbol;EMA_PERIOD;EMA_PROFIT;EMA_OPS;EMA_COST;EMA_NET"
        print(header)
        count=1
        sma_best_period = 0
        for symbol in self.get_symbols():
            if symbol != "BTCEUR":
               continue
            try:
                candlesticks = self.get_candlesticks(symbol, days, "4h")
                closes = self.get_close_array_from_candlesticks(candlesticks)
                if not closes:
                    continue
                close_avg = np.mean(closes)
                # sma_best_period, sma_best_profit, sma_best_cost = self.optimize_params(candlesticks,"sma")
                # smma_best_period, smma_best_profit, smma_best_ops = self.optimize_params(candlesticks,"smma")
                ema_profit, ema_ops, ema_cost = self.cross_ema_stragegy(candlesticks)
                print(f"{count};{symbol};EMA_9_26;{ema_profit};{ema_ops};{ema_cost};{ema_profit-ema_cost}")
            except Exception as e:
                print(e)
                continue
            count+=1

if __name__ == "__main__":
    trade = Trade()
    trade.simulate()
