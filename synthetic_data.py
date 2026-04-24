import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_candles(
    n_candles: int = 500,
    base_price: float = 100.0,
    timeframe_minutes: int = 5,
    start_time: datetime = None,
    wave_period_range: tuple = (50, 200),
    amplitude_range: tuple = (5.0, 25.0),
    amplitude_change_speed: float = 0.02,
    candle_noise_pct: float = 0.005,
    wick_ratio_range: tuple = (0.3, 1.5),
    trend_strength: float = 0.0,
    random_seed: int = None
) -> pd.DataFrame:
    """
    Szintetikus OHLCV gyertyás adatokat generál változó amplitúdójú szinusz hullám alapján.

    Paraméterek
    -----------
    n_candles             : Generálandó gyertyák száma (alapértelmezett: 500)
    base_price            : Kiindulóár (alapértelmezett: 100.0)
    timeframe_minutes     : Gyertya időkeret percben (alapértelmezett: 5)
    start_time            : Kezdő timestamp; None = 2020-01-01 00:00
    wave_period_range     : (min, max) szinusz periódus gyertyában (alapértelmezett: 50–200)
    amplitude_range       : (min, max) amplitúdó ár-egységben (alapértelmezett: 5–25)
    amplitude_change_speed: Amplitúdó véletlen bolyongás sebessége – minél nagyobb,
                            annál gyorsabban változik az amplitúdó (alapértelmezett: 0.02)
    candle_noise_pct      : Open/Close véletlen eltérés a centerárhoz képest, arányban
                            (pl. 0.005 = ±0.5%) (alapértelmezett: 0.005)
    wick_ratio_range      : (min, max) kanóc/test arány tartomány (alapértelmezett: 0.3–1.5)
    trend_strength        : Gyertyánkénti lineáris trend ár-egységben
                            (pl. 0.05 = lassú emelkedés, -0.05 = csökkenés) (alapértelmezett: 0.0)
    random_seed           : Reprodukálhatósághoz; None = véletlenszerű (alapértelmezett: None)

    Visszatérési érték
    ------------------
    pd.DataFrame oszlopokkal:
        time  – gyertya nyitó ideje
        open       – nyitóár
        high       – maximum ár
        low        – minimum ár
        close      – záróár
        volume     – kereskedési volumen

    Megjegyzés
    ----------
    A szinusz hullám mind az amplitúdója, mind a periódusa lassan változik (véletlen
    bolyongás log-térben), így az eredmény valósabb, mint egy fix paraméterű hullám.
    RL tanítóadatként ajánlott clean verzió (debug oszlopok nélkül) exportálni.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if start_time is None:
        start_time = datetime(2020, 1, 1)

    t = np.arange(n_candles)

    # --- Változó amplitúdó (véletlen bolyongás log-térben) ---
    log_amp = np.zeros(n_candles)
    log_amp[0] = np.random.uniform(np.log(amplitude_range[0]), np.log(amplitude_range[1]))
    amp_steps = np.random.normal(0, amplitude_change_speed, n_candles - 1)
    for i in range(1, n_candles):
        log_amp[i] = np.clip(
            log_amp[i-1] + amp_steps[i-1],
            np.log(amplitude_range[0]),
            np.log(amplitude_range[1])
        )
    amplitudes = np.exp(log_amp)

    # --- Változó periódus (lassan változó frekvencia) ---
    log_period = np.zeros(n_candles)
    log_period[0] = np.random.uniform(np.log(wave_period_range[0]), np.log(wave_period_range[1]))
    period_steps = np.random.normal(0, 0.01, n_candles - 1)
    for i in range(1, n_candles):
        log_period[i] = np.clip(
            log_period[i-1] + period_steps[i-1],
            np.log(wave_period_range[0]),
            np.log(wave_period_range[1])
        )
    periods = np.exp(log_period)

    # --- Fázis integrálása (változó frekvenciából fázisszög) ---
    phase = np.zeros(n_candles)
    for i in range(1, n_candles):
        phase[i] = phase[i-1] + (2 * np.pi / periods[i-1])

    # --- Középár (szinusz + lineáris trend + kis véletlen bolyongás) ---
    trend = trend_strength * t
    random_walk = np.cumsum(np.random.normal(0, base_price * 0.001, n_candles))
    center_price = base_price + trend + amplitudes * np.sin(phase) + random_walk
    center_price = np.maximum(center_price, base_price * 0.1)

    # --- OHLC generálás ---
    open_noise  = np.random.normal(0, center_price * candle_noise_pct)
    close_noise = np.random.normal(0, center_price * candle_noise_pct)
    open_prices  = center_price + open_noise
    close_prices = center_price + close_noise

    body_size  = np.abs(close_prices - open_prices)
    wick_ratio = np.random.uniform(wick_ratio_range[0], wick_ratio_range[1], n_candles)
    wick_size  = body_size * wick_ratio + center_price * 0.001  # minimális kanóc

    upper_frac  = np.random.uniform(0, 1, n_candles)
    upper_wick  = upper_frac * wick_size
    lower_wick  = (1 - upper_frac) * wick_size

    high_prices = np.maximum(open_prices, close_prices) + upper_wick
    low_prices  = np.minimum(open_prices, close_prices) - lower_wick
    low_prices  = np.maximum(low_prices, 0.01)

    # --- Volume (amplitúdóval arányos + log-normális zaj) ---
    volume = (
        1000
        * (1 + amplitudes / amplitude_range[1])
        * np.random.lognormal(0, 0.4, n_candles)
    ).astype(int)

    timestamps = [start_time + timedelta(minutes=timeframe_minutes * i) for i in range(n_candles)]

    return pd.DataFrame({
        "time": timestamps,
        "open":      np.round(open_prices,  4),
        "high":      np.round(high_prices,  4),
        "low":       np.round(low_prices,   4),
        "close":     np.round(close_prices, 4),
        "volume":    volume,
    })


# ---------------------------------------------------------------------------
# Példa használat
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_synthetic_candles(
        n_candles=1000,
        base_price=100.0,
        timeframe_minutes=5,
        wave_period_range=(50, 150),
        amplitude_range=(3.0, 20.0),
        amplitude_change_speed=0.03,
        candle_noise_pct=0.004,
        trend_strength=0.0,
        random_seed=42
    )
    print(df.head())
    print(f"Ár tartomány: {df['low'].min():.2f} – {df['high'].max():.2f}")
    df.to_csv("synthetic_candles.csv", index=False)