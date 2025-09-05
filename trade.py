import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
from binance.client import Client
import mplfinance as mpf
import yfinance as yf


class Trade:
    def __init__(self):
        pass

    def get_currency_list(self):
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        return data

    def get_euro_based_currency_list(self):
        data = self.get_currency_list()
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


if __name__ == "__main__":
    trade = Trade()