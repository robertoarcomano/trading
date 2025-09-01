from binance.client import Client
import pandas as pd
import datetime

def get_values():
    days=1
    # API key e secret non necessarie per dati pubblici
    client = Client()

    # Parametri
    symbol = "SOLEUR"
    interval = Client.KLINE_INTERVAL_15MINUTE

    # Calcola la data di inizio (2 mesi fa)
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days)

    # Recupera i dati delle candele (klines)
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_time.strftime("%d %b, %Y %H:%M:%S"),
        end_time.strftime("%d %b, %Y %H:%M:%S")
    )

    # Organizza i dati in DataFrame per comodità
    df = pd.DataFrame(klines, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])

    # Converti timestamp in formato leggibile per 'Open time'
    df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')

    # Imposta indice temporale
    df.set_index("Open time", inplace=True)

    # Converti colonne numeriche
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Visualizza le prime righe
    # print(df.head())
    close_series = df["Close"]

    values = [float(rec[3]) for rec in klines]
    # for rec in klines:
        # print(float(rec[3]))
    # print(values)

    # Opzionale: salva dati su CSV
    df.to_csv("solana_eur_15m_last_2months.csv")
    return values
        