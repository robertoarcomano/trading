import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
from binance.client import Client
import mplfinance as mpf
import yfinance as yf


class Trade:
    # Binance applies a 0.1% fee for every trade
    # And a 25% discount if use BNB to pay the fee
    # BINANCE_OP_COST = 0.075
    BINANCE_OP_COST = 0.1
    def __init__(self, show_plots=False, show_trades=False, show_data=False, ignore_trends=False, use_volume=False):
        self.show_plots = show_plots
        self.show_trades = show_trades 
        self.show_data = show_data 
        self.ignore_trends = ignore_trends
        self.use_volume = use_volume

    def set_ignore_trends(self, ignore_trends):
        self.ignore_trends = ignore_trends  

    def set_use_volume(self, use_volume):
        self.use_volume = use_volume

    def get_symbols(self):
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()

        symbols_eur = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'EUR']
        return symbols_eur

    def get_candlesticks(self, symbol, start, days, interval):
        # API key e secret non necessarie per dati pubblici
        client = Client()

        # Parametri
        start_time = datetime.datetime.strptime(start, "%m-%Y")
        end_time = start_time + datetime.timedelta(days=days)
        s = start_time.strftime("%d %b, %Y %H:%M:%S")
        e = end_time.strftime("%d %b, %Y %H:%M:%S")
        # Recupera i dati delle candele (klines)
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_time.strftime("%d %b, %Y %H:%M:%S"),
            end_time.strftime("%d %b, %Y %H:%M:%S")
        )
        open_prices = [float(rec[1]) for rec in klines]
        close_prices = [float(rec[4]) for rec in klines]        
        return [[float(rec[1]),float(rec[4]),float(rec[5])] for rec in klines]

    def do_calculate_ema(self, array, period):
        ema = []
        k = 2 / (period + 1)
        sma = sum(array[:period]) / period
        ema.append(sma)

        for price in array[period:]:
            ema_value = (price * k) + (ema[-1] * (1 - k))
            ema.append(ema_value)
        
        ema = [np.nan]*(period-1) + ema
        return ema        

    def calculate_ema(self, candlesticks, period):
        closes = self.get_close_array_from_candlesticks(candlesticks)
        return self.do_calculate_ema(closes, period)

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

    # def cross_ema_strategy(self, candlesticks, low, high, start_time, interval):
    #     ema_low = self.calculate_ema(candlesticks, low)
    #     ema_high = self.calculate_ema(candlesticks, high)    
    #     period = max(low, high)
    #     position = 0
    #     cash = 0
    #     entry_price = 0
    #     cost = 0
    #     ops = 0
    #     for i in range(len(candlesticks)):
    #         candlestick = candlesticks[i]
    #         open = candlestick[0]
    #         close = candlestick[1]
    #         if close < open:
    #             type_trade = "red"
    #         elif close > open:
    #             type_trade = "green"
    #         else:
    #             type_trade = "equal"
    #         if position == 0:
    #             if already_crossed and type_trade == "green" and ema_low[i] > ema_high[i] and i > period:
    #                 if self.show_trades:
    #                     print(i,type_trade,"BUY:",candlestick, ema_low[i], ema_high[i], cash, cost)
    #                 position = 1
    #                 entry_price = close
    #                 cost += close * Trade.BINANCE_OP_COST / 100
    #                 ops += 1
                
    #         elif position == 1:
    #             if type_trade == "red" and ema_low[i] < ema_high[i] and i > period:
    #                 if self.show_trades:
    #                     print(i, type_trade,"SELL:",candlestick, ema_low[i], ema_high[i], cash, cost)
    #                 cash += close - entry_price
    #                 position = 0
    #                 cost += close * Trade.BINANCE_OP_COST / 100
    #                 ops += 1

    #     if position == 1:
    #         cash += candlesticks[-1][1] - entry_price
    #         cost += close * Trade.BINANCE_OP_COST / 100
    #         ops += 1
    #     precision = 4
    #     avg = self.get_average_from_candlesticks(candlesticks)
    #     if self.show_trades:
    #         print("cash:", round(cash,precision), "cost:", round(cost,precision), "net profit:", round(cash - cost,precision), "net profit percentage:", round((cash - cost)*100/avg,precision))
    #     if self.show_plots:
    #         self.plot_candlesticks(candlesticks, ema_low, ema_high, start_time, interval) 
    #     return round(cash,precision), ops, round(cost/ops,precision) if ops>0 else 0, round(cost,precision), round(cash - cost,precision), round((cash - cost)*100/avg,precision)     
    def cross_ema_strategy(self, candlesticks, low, high, start_time, interval, long=None):
        ema_low = self.calculate_ema(candlesticks, low)
        ema_high = self.calculate_ema(candlesticks, high)    
        ema_long = self.calculate_ema(candlesticks, long) if long is not None else None
        volume_avg = self.do_calculate_ema([candlestick[2] for candlestick in candlesticks], 20)
        period = max(low, high)
        
        position = 0
        cash = 0
        entry_price = 0
        cost = 0
        ops = 0
        
        # stato iniziale: None = sconosciuto, 1 = ema_low > ema_high, -1 = ema_low < ema_high
        prev_state = None

        for i in range(len(candlesticks)):
            candlestick = candlesticks[i]
            open_price = candlestick[0]
            close_price = candlestick[1]
            volume = candlestick[2]

            if close_price < open_price:
                type_trade = "red"
            elif close_price > open_price:
                type_trade = "green"
            else:
                type_trade = "equal"

            # stato corrente EMA
            if ema_low[i] > ema_high[i]:
                state = 1
            elif ema_low[i] < ema_high[i]:
                state = -1
            else:
                state = prev_state  # se uguali mantieni lo stato precedente

            # crossover rilevato solo se cambia lo stato
            crossed = prev_state is not None and state != prev_state
            prev_state = state

            if position == 0:
                if crossed and ema_low[i] > ema_high[i] and i > period and (self.ignore_trends or type_trade == "green") and (ema_long is None or close_price > ema_long[i]) and \
                    (not self.use_volume or volume > volume_avg[i]):
                    if self.show_trades:
                        print(i, type_trade, "BUY:", candlestick, ema_low[i], ema_high[i], cash, cost)
                    position = 1
                    entry_price = close_price
                    cost += close_price * Trade.BINANCE_OP_COST / 100
                    ops += 1

            elif position == 1:
                if crossed and ema_low[i] < ema_high[i] and i > period and (self.ignore_trends or type_trade == "red") and (ema_long is None or close_price < ema_long[i]) and \
                    (not self.use_volume or volume > volume_avg[i]):
                    if self.show_trades:
                        print(i, type_trade, "SELL:", candlestick, ema_low[i], ema_high[i], cash, cost)
                    cash += close_price - entry_price
                    position = 0
                    cost += close_price * Trade.BINANCE_OP_COST / 100
                    ops += 1

        # chiudi eventuale posizione aperta
        if position == 1:
            if self.show_trades:
                print(i, type_trade, "SELL:", candlesticks[-1], ema_low[-1], ema_high[-1], cash, cost)
            cash += candlesticks[-1][1] - entry_price
            cost += close_price * Trade.BINANCE_OP_COST / 100
            ops += 1

        precision = 4
        avg = self.get_average_from_candlesticks(candlesticks)
        if self.show_trades:
            print("cash:", round(cash, precision), 
                "cost:", round(cost, precision), 
                "net profit:", round(cash - cost, precision), 
                "net profit percentage:", round((cash - cost)*100/avg, precision))
        if self.show_plots:
            self.plot_candlesticks(candlesticks, ema_low, ema_high, start_time, interval)

        return (round(cash, precision), 
                ops, 
                round(cost/ops, precision) if ops>0 else 0, 
                round(cost, precision), 
                round(cash - cost, precision), 
                round((cash - cost)*100/avg, precision))

    def net_profit(self, best_profit, avg, days, best_ops):
        return f"{round(best_profit*100/avg/days-best_ops/days*Trade.BINANCE_OP_COST,1)}"    
        # return f"{round(best_profit*100/avg/days,1)}"    

    @staticmethod
    def get_close_array_from_candlesticks(candlesticks): 
        return [candlestick[1] for candlestick in candlesticks]

    @staticmethod
    def get_average_from_candlesticks(candlesticks): 
        return np.mean([candlestick[1] for candlestick in candlesticks])

    @staticmethod
    def get_average_array_from_candlesticks(candlesticks): 
        return [np.mean(candlestick) for candlestick in candlesticks]

    def plot_candlesticks(self, candlesticks, ema1, ema2, start_time, interval):
        df = pd.DataFrame(candlesticks, columns=['Open', 'Close'])
        df['High'] = df[['Open', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Close']].min(axis=1)
        df.index = pd.date_range(start=start_time, periods=len(df), freq=interval)

        ap1 = mpf.make_addplot(ema1, panel=0, color='blue')
        ap2 = mpf.make_addplot(ema2, panel=0, color='green')

        mpf.plot(df, type='candle', style='charles', addplot=[ap1, ap2], title='BTCEUR Candlestick', volume=False, warn_too_much_data=100000000)

    def backtest_cross_ema_strategy(self, symbol, start_time=(datetime.datetime.now()- datetime.timedelta(days=30)).strftime("%m-%Y"), days=30, interval="4h", low=9, high=26, long=None):
        candlesticks = self.get_candlesticks(symbol, start_time, days, interval)
        if self.show_data:
            ema_low = self.calculate_ema(candlesticks, low)
            ema_high = self.calculate_ema(candlesticks, high)    
            for i in range(0, len(candlesticks)): 
                print(i,candlesticks[i], ema_low[i], ema_high[i])
        profit, ops, single_cost, cost, net_profit, net_profit_percentage = self.cross_ema_strategy(candlesticks, low, high, start_time, interval, long)
        return net_profit_percentage

    def simulate(self):
        days = 30
        delta_max = 30
        header = "Symbol;PERIOD;PROFIT;OPS;SINGLE_COST;COST;NET_PROFIT;NET_PROFIT_PERCENTAGE"
        print(header)
        count=1
        symbol_array = []
        universal_array = {}
        for delta in range(0, delta_max):
            for symbol in self.get_symbols():
                if symbol != "BTCEUR":
                    continue
                try:
                    candlesticks = self.get_candlesticks(symbol, days, "4h", delta)
                    closes = self.get_close_array_from_candlesticks(candlesticks)
                    if not closes:
                        continue
                    avg = np.mean(closes)
                    best_net_profit = 0
                    best_low_period = 0
                    best_high_period = 0
                    for high_period in range(10, round(days*6)):
                        for low_period in range(1, round(days*6)):
                            if low_period >= high_period:
                                continue    
                            ema_low = self.calculate_ema(candlesticks, low_period)
                            ema_high = self.calculate_ema(candlesticks, high_period)
                            profit, ops, single_cost, cost, net_profit, p = self.cross_ema_strategy(candlesticks, ema_low, ema_high, max(low_period, high_period), debug=False)
                            if net_profit > best_net_profit:
                                best_net_profit = net_profit
                                best_ops = ops
                                best_single_cost = single_cost
                                best_cost = cost
                                best_profit = profit 
                                best_low_period = low_period
                                best_high_period = high_period       
                            key = "EMA" + "_" + str(low_period) + "_" + str(high_period)
                            if key not in universal_array:
                                universal_array[key] = ([delta, symbol, "EMA" + "_" + str(low_period) + "_" + str(high_period), profit, ops, single_cost, cost, net_profit, float(round(100*net_profit/avg,2))])
                            else:
                                new_item = universal_array[key]
                                new_item[8] = round( (new_item[8] + float(round(100*net_profit/avg,2)))/2,2)
                                universal_array[key] = new_item
                            # self.plot_candlesticks(candlesticks, ema_low, ema_high)
                            # print(f"{count};{symbol};EMA_{low_period}_{high_period};{profit};{ops};{single_cost};{cost};{net_profit}")
                    if best_low_period == 0 or best_high_period == 0:
                        continue   
                    # print(f"{count};{symbol};EMA_{best_low_period}_{best_high_period};{best_profit};{best_ops};{best_single_cost};{best_cost};{best_net_profit};{round(100*best_net_profit/avg,1)}%")
                    symbol_array.append([delta, symbol, "EMA" + "_" + str(best_low_period) + "_" + str(best_high_period), best_profit, best_ops, best_single_cost, best_cost, best_net_profit, float(round(100*best_net_profit/avg,2))])     
                except Exception as e:
                    print(e)
                    continue
                count+=1
                # if count > 5:
                    # break
        top_3 = sorted(universal_array.items(), key=lambda item: item[1][8], reverse=True)[:3]
        for key, values in top_3:
            print(f"{key}: {values[8]}")
        symbol_array.sort(key=lambda x: x[8], reverse=True)
        for symbol in symbol_array:
            print(f"{symbol[0]};{symbol[1]};{symbol[2]};{symbol[3]};{symbol[4]};{symbol[5]};{symbol[6]};{symbol[7]};{symbol[8]}%")

if __name__ == "__main__":
    trade = Trade(show_plots=False, show_trades=False, show_data=False)
    net_profit_percentage_array = []
    symbol = "SOLEUR"
    net_profit_arr = []
    net_profit_long_arr = []
    for month in range(1,13):
        # net_profit_percentage_array.append(float(trade.backtest_cross_ema_strategy("BTCEUR", str(month).zfill(2) + "-2024", 30, low=5)))
        a = []
        for use_volume in [False, True]:
            trade.set_use_volume(use_volume)
            a.append(float(trade.backtest_cross_ema_strategy(symbol, str(month).zfill(2) + "-2023", 30, low=9)))
        print(a)
        # net_profit_long_arr.append(float(trade.backtest_cross_ema_strategy(symbol, str(month).zfill(2) + "-2023", 30, low=9, long=200)))
    # print(np.mean(net_profit_arr), np.mean(net_profit_long_arr))
    # print(net_profit_percentage_array)
    # avg, std = round(np.mean(net_profit_percentage_array),2), round(np.std(net_profit_percentage_array),2)
    # print("net_profit:",avg,"% ±",std,"%")