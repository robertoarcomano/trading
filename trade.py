import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
from binance.client import Client
import mplfinance as mpf
import yfinance as yf
import json


class Trade:
    def __init__(self):
        self.candlestick_filename = "btceur_2025_15m.json"
        self._show_trades = False
        # Binance applies a 0.1% fee for every trade
        # And a 25% discount if use BNB to pay the fee
        # BINANCE_OP_COST = 0.075
        self.BINANCE_OP_COST = 0.1
        self.BINANCE_OP_MOLT = self.BINANCE_OP_COST / 100

    @property
    def show_trades(self):    
        return self._show_trades

    @show_trades.setter
    def show_trades(self, value):
        self._show_trades = value

    def get_currency_list(self):
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        return data

    def get_euro_based_currency_list(self):
        data = self.get_currency_list()
        symbols_eur = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'EUR']
        return symbols_eur

    def get_candlesticks(self, symbol, start, stop, interval):
        client = Client()

        start_time = datetime.datetime.strptime(start, "%m-%Y")
        end_time = datetime.datetime.strptime(stop, "%m-%Y")
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_time.strftime("%d %b, %Y %H:%M:%S"),
            end_time.strftime("%d %b, %Y %H:%M:%S")
        )
        # Open, Close, Volume
        self.candlesticks = [[float(rec[1]),float(rec[4]),float(rec[5])] for rec in klines]

    def load_candlesticks(self):
        with open(self.candlestick_filename, "r") as f:
            self.candlesticks = json.load(f)

    def save_candlesticks(self):
        with open(self.candlestick_filename, "w") as f:
            json.dump(self.candlesticks, f)

    def get_average_close_from_candlesticks(self): 
        return np.mean([candlestick[1] for candlestick in self.candlesticks])

    def calculate_ema(self, period):
        data = [candlestick[1] for candlestick in self.candlesticks]
        ema = []
        k = 2 / (period + 1)
        sma = sum(data[:period]) / period
        ema.append(sma)

        for price in data[period:]:
            ema_value = (price * k) + (ema[-1] * (1 - k))
            ema.append(ema_value)
        
        ema = [np.nan]*(period-1) + ema
        return ema   

    def build_emas(self, short_period, medium_period, long_period):
        self.build_ema_short(short_period)
        self.build_ema_medium(medium_period)
        self.build_ema_long(long_period)

    def build_ema_short(self, period):
        self.ema_short, self.ema_short_period = self.calculate_ema(period), period

    def build_ema_medium(self, period):
        self.ema_medium, self.ema_medium_period = self.calculate_ema(period), period

    def build_ema_long(self, period):
        self.ema_long, self.ema_long_period = self.calculate_ema(period), period
        
    def get_max_period(self):
        return max(self.ema_short_period, self.ema_medium_period, self.ema_long_period) 

    def backtest_cross_ema_strategy(self):
        position = 0
        cash = 0
        entry_price = 0
        cost = 0
        ops = 0
        prev_state = None

        for i in range(self.get_max_period(), len(self.candlesticks)):
            candlestick = self.candlesticks[i]
            open_price, close_price, volume = candlestick[0], candlestick[1], candlestick[2]

            if close_price < open_price:
                type_trade = "red"
            elif close_price > open_price:
                type_trade = "green"
            else:
                type_trade = "equal"

            if self.ema_short[i] > self.ema_medium[i]:
                state = 1
            elif self.ema_short[i] < self.ema_medium[i]:
                state = -1
            else:
                state = prev_state

            crossed = prev_state is not None and state != prev_state
            prev_state = state

            if position == 0:
                if crossed and self.ema_short[i] > self.ema_medium[i] and (self.ema_long_period == 0 or close_price > self.ema_long[i]):
                    # and (self.ignore_trends or type_trade == "green") and
                    # (not self.use_volume or volume > volume_avg[i])
                    if self.show_trades:
                        print(i, type_trade, "BUY:", candlestick, self.ema_short[i], self.ema_medium[i], cash, cost)
                    position = 1
                    entry_price = close_price
                    cost += close_price * self.BINANCE_OP_MOLT
                    ops += 1

            elif position == 1:
                if crossed and self.ema_short[i] < self.ema_medium[i] and (self.ema_long_period == 0 or close_price < self.ema_long[i]):
                    # and (self.ignore_trends or type_trade == "red")
                    # (not self.use_volume or volume > volume_avg[i]):
                    if self.show_trades:
                        print(i, type_trade, "SELL:", candlestick, self.ema_short[i], self.ema_medium[i], cash, cost)
                    cash += close_price - entry_price
                    position = 0
                    cost += close_price * self.BINANCE_OP_MOLT
                    ops += 1

        if position == 1:
            if self.show_trades:
                print(i, type_trade, "SELL:", candlesticks[-1], self.ema_short[-1], self.ema_medium[-1], cash, cost)
            cash += self.candlesticks[-1][1] - entry_price
            cost += close_price * self.BINANCE_OP_MOLT
            ops += 1

        precision = 4
        average_close = self.get_average_close_from_candlesticks()
        if self.show_trades:
            print("cash:", round(cash, precision), 
                "cost:", round(cost, precision), 
                "net profit:", round(cash - cost, precision), 
                "net profit percentage:", round((cash - cost)*100/average_close, precision))
        # if self.show_plots:
            # self.plot_candlesticks(candlesticks, ema_low, ema_high, start_time, interval)

        return (float(round(cash, precision)), 
                ops, 
                float(round(cost/ops, precision)) if ops>0 else 0, 
                float(round(cost, precision)), 
                float(round(cash - cost, precision)), 
                float(round((cash - cost)*100/average_close, precision)))


if __name__ == "__main__":
    trade = Trade()
    trade.load_candlesticks()
    trade.build_emas(9,26,200)
    trade.show_trades = True
    profit = trade.backtest_cross_ema_strategy()
    print(profit)
    