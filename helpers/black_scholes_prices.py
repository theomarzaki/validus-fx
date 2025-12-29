import numpy as np
from helpers.delta_strikes import calculate_25delta_strikes
from scipy.stats import norm


def black_scholes_price(S, K, T, sigma, r, q, option_type="call"):
    """Calculate Black-Scholes option price"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


def calculate25DeltaVolatilities(dataset, year):
    latest_data = dataset.iloc[-1]
    sigma_atm = latest_data[f"EURUSD_{year}Y_ATM_VOL_MID"]
    rr = latest_data[f"EURUSD_{year}Y_25DELTA_Risk_Reversal_MID"]
    bf = latest_data[f"EURUSD_{year}Y_25DELTA_Butterfly_MID"]

    sigma_25d_call = sigma_atm + bf + rr / 2
    sigma_25d_put = sigma_atm + bf - rr / 2

    return sigma_atm, sigma_25d_call, sigma_25d_put


def getBlackScholesOptions(
    dataset,
    params,
    year,
):

    latest_data = dataset.iloc[-1]

    sigma_atm, sigma_25d_call, sigma_25d_put = calculate25DeltaVolatilities(
        dataset=dataset, year=year
    )

    K_call, K_put = calculate_25delta_strikes(
        params["S0"],
        year,
        sigma_atm,
        sigma_25d_call,
        sigma_25d_put,
        params["usd_ir"],
        params["eur_ir"],
    )

    F = params["S0"] * np.exp((params["usd_ir"] - params["eur_ir"]) * year)

    price_atm_mkt = black_scholes_price(
        params["S0"], F, year, sigma_atm, params["usd_ir"], params["eur_ir"], "call"
    )
    price_call_mkt = black_scholes_price(
        params["S0"],
        K_call,
        year,
        sigma_25d_call,
        params["usd_ir"],
        params["eur_ir"],
        "call",
    )
    price_put_mkt = black_scholes_price(
        params["S0"],
        K_put,
        year,
        sigma_25d_put,
        params["usd_ir"],
        params["eur_ir"],
        "put",
    )

    return F, K_call, K_put, price_atm_mkt, price_call_mkt, price_put_mkt
