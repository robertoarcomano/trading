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
        self.candelsticks = [[float(rec[1]),float(rec[4]),float(rec[5])] for rec in klines]

    def load_candlesticks(self):
        with open(self.candlestick_filename, "r") as f:
            self.candelsticks = json.load(f)

    def save_candlesticks(self):
        with open(self.candlestick_filename, "w") as f:
            json.dump(self.candelsticks, f)

    def calculate_ema(self, period):
        data = [candlestick[1] for candlestick in self.candelsticks]
        ema = []
        k = 2 / (period + 1)
        sma = sum(data[:period]) / period
        ema.append(sma)

        for price in data[period:]:
            ema_value = (price * k) + (ema[-1] * (1 - k))
            ema.append(ema_value)
        
        ema = [np.nan]*(period-1) + ema
        return ema   

    def build_emas(self, ema_short, ema_medium, ema_long):
        self.ema_short = self.calculate_ema(ema_short)
        self.ema_medium = self.calculate_ema(ema_medium)
        self.ema_long = self.calculate_ema(ema_long)

    def build_ema_short(self, ema_short):
        self.ema_short = self.calculate_ema(ema_short)

    def build_ema_medium(self, ema_medium):
        self.ema_medium = self.calculate_ema(ema_medium)

    def build_ema_long(self, ema_long):
        self.ema_long = self.calculate_ema(ema_long)
        
        
if __name__ == "__main__":
    trade = Trade()
    trade.load_candlesticks()
    trade.build_emas(9,26,100)
