import pandas as pd
from model.Heston import HestonModel
import numpy as np
from helpers.black_scholes_prices import getBlackScholesOptions
from pathlib import Path
from model.test.TestHeston import TestHestonModel
import pickle
from datetime import datetime, timedelta
from helpers.date_sampler import sampleAtKeyDates
from strategies.StaticForward import StaticForwardHedging
from strategies.PartialHedge import PartialForwardHedging
from strategies.DynamicDelta import DynamicDeltaHedging
from strategies.NoHedging import NoHedging
from metrics.IRR import calculate_irr
from metrics.MultipleCapital import calculate_multiple_on_capital
from metrics.VAR import calculate_cvar, calculate_var
from metrics.ExtremeScenarios import calculate_extreme_scenarios
from plotter.StrategyCompare import plotComparisons, plotExtremes
from metrics.CostBenefitAnalysis import calculate_cost_benefit_analysis

DEBUG = False
PLOT = False

cash_flows_eur = {
    "2025-10-01": -10000000,  # Initial investment (outflow)
    "2026-10-01": 1000000,  # Year 1
    "2027-10-01": 1000000,  # Year 2
    "2029-10-01": 1000000,  # Year 3
    "2030-10-01": 11000000,  # Year 4
}


def getDataset():
    return pd.read_csv("data/processed_data.csv")


def getInitialParameters(dataset):
    latest_data = dataset.iloc[-1]

    """ Calculate Current Data """
    mu = dataset["EURUSD_Spot_LOG_RETURNS"].mean() * 252
    v0 = (latest_data["EURUSD_1Y_ATM_VOL_MID"]) ** 2
    theta = (latest_data["EURUSD_5Y_ATM_VOL_MID"]) ** 2

    dataset["EURUSD_REALISED_VOL_21D"] = (
        dataset["EURUSD_Spot_LOG_RETURNS"].rolling(21).std() * np.sqrt(252)
    ) / 100
    realised_vol = dataset["EURUSD_REALISED_VOL_21D"].dropna()
    kappa = 1.5
    if len(realised_vol) > 1:
        autocorr = realised_vol.autocorr(lag=1)
        kappa = -np.log(autocorr) * 252 if autocorr > 0 else 1.5

    implied_vol = (
        dataset["EURUSD_1Y_ATM_VOL_MID"].pct_change().std() * np.sqrt(252)
    ) / 100
    sigma = implied_vol if not np.isnan(implied_vol) else 0.3
    rho = -0.4

    initial_params = {
        "S0": latest_data["EURUSD_Spot_MID"],
        "v0": v0,
        "theta": theta,
        "kappa": kappa,
        "sigma": sigma,
        "rho": rho,
        "mu": mu,
        "usd_ir": 0.035,
        "eur_ir": 0.0215,
    }

    return initial_params


def getHistoricalStats(dataset):
    log_returns = dataset["EURUSD_Spot_LOG_RETURNS"]
    historical_stats = {
        "Annualized Mean": log_returns.mean() * 252,
        "Annualized Vol": log_returns.std() * np.sqrt(252),
        "Skewness": log_returns.skew(),
        "Kurtosis": log_returns.kurtosis(),
        "Current 1Y IV": latest_data["EURUSD_1Y_ATM_VOL_MID"],
        "Current 5Y IV": latest_data["EURUSD_5Y_ATM_VOL_MID"],
    }
    return historical_stats


def getKeyDates():
    last_date = datetime(2025, 8, 1)  # last date in the dataset
    cash_flow_dates = [
        datetime.strptime(date, "%Y-%m-%d") for date in cash_flows_eur.keys()
    ]
    times_to_cf = [(date - last_date).days / 365.25 for date in cash_flow_dates]

    return cash_flow_dates, times_to_cf


def getForwardRates(initial_params):
    forward_rates = {}
    cash_flow_dates, times_to_cf = getKeyDates()
    for date, T in zip(cash_flow_dates, times_to_cf):
        forward_rate = initial_params["S0"] * np.exp(
            (initial_params["usd_ir"] - initial_params["eur_ir"]) * T
        )
        forward_rates[date] = forward_rate
    return forward_rates


def buildAndCalibrateModel(dataset, initial_params):

    model = HestonModel(S0=initial_params["S0"], params=initial_params)

    F_1y, K_call_1y, K_put_1y, price_atm_1y_mkt, price_call_1y_mkt, price_put_1y_mkt = (
        getBlackScholesOptions(dataset, initial_params, 1)
    )
    F_5y, K_call_5y, K_put_5y, price_atm_5y_mkt, price_call_5y_mkt, price_put_5y_mkt = (
        getBlackScholesOptions(dataset, initial_params, 5)
    )

    market_data = {
        "F_1y": F_1y,
        "K_call_1y": K_call_1y,
        "K_put_1y": K_put_1y,
        "F_5y": F_5y,
        "K_call_5y": K_call_5y,
        "K_put_5y": K_put_5y,
        "price_atm_1y_mkt": price_atm_1y_mkt,
        "price_call_1y_mkt": price_call_1y_mkt,
        "price_put_1y_mkt": price_put_1y_mkt,
        "price_atm_5y_mkt": price_atm_5y_mkt,
        "price_call_5y_mkt": price_call_5y_mkt,
        "price_put_5y_mkt": price_put_5y_mkt,
    }

    params_file = "model/calibrated_heston_params.pkl"
    if Path(params_file).is_file():
        with open(params_file, "rb") as file:
            params = pickle.load(file)
            model.set_parameters(params)
    else:
        model.calibrate(market_data)

    if DEBUG:
        model_tester = TestHestonModel(model, market_data, initial_params)
        model_tester.visualiseSpotPaths()
        model_tester.visualiseVolPaths()
        model_tester.visualiseTerminalDistribution()
        model_tester.visualiseVolatilitySmile()

    return model


if __name__ == "__main__":

    dataset = getDataset()
    initial_params = getInitialParameters(dataset)

    model = buildAndCalibrateModel(dataset, initial_params)

    cash_flow_dates, times_to_cf = getKeyDates()
    T_horizon = max(times_to_cf)

    t, S, v = model.simulate(T_horizon)

    spot_at_cf_dates, vol_at_cf_dates = sampleAtKeyDates(t, S, v, times_to_cf)
    forward_rates = getForwardRates(initial_params)

    NoStrategy = NoHedging(cash_flows_eur)
    staticHedging = StaticForwardHedging(cash_flows_eur)
    partialHedging = PartialForwardHedging(cash_flows_eur)
    partialHedging.name = "Partial Hedge 0.5"
    partialHedging_8 = PartialForwardHedging(cash_flows_eur)
    partialHedging_8.hedge_ratio = 0.8
    partialHedging_8.name = "Partial Hedge 0.8"
    dynamicHedging = DynamicDeltaHedging(cash_flows_eur)

    strategies = [
        NoStrategy,
        staticHedging,
        partialHedging,
        partialHedging_8,
        dynamicHedging,
    ]
    results = []

    for strategy in strategies:
        usd_cf = strategy.calculate_usd_cf(spot_at_cf_dates, forward_rates)

        irr = calculate_irr(usd_cf, times_to_cf)
        multiples = calculate_multiple_on_capital(usd_cf)
        var = calculate_var(irr)
        cvar = calculate_cvar(irr)

        result = {
            "Strategy Name": strategy.name,
            "IRR": irr,
            "Multiples": multiples,
            "VaR": var,
            "CVaR": cvar,
            "USD_CF": usd_cf,
        }

        results.append(result)

    results = pd.DataFrame(results)

    extreme_scenario = []
    for index, result in results.iterrows():
        extreme_scenario.append(calculate_extreme_scenarios(result))
    extreme_scenario = pd.DataFrame(extreme_scenario)

    if PLOT:
        plotComparisons(results)
        plotExtremes(extreme_scenario)

    cost_benefit_analysis = calculate_cost_benefit_analysis(results)

    print(cost_benefit_analysis)
