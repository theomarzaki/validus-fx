from strategies.Hedging import HedgingStrategy
import numpy as np


class DynamicDeltaHedging(HedgingStrategy):
    def __init__(self, cash_flows_eur):
        super().__init__("Dynamic Delta Hedge")
        self.cash_flows_eur = cash_flows_eur

    def calculate_usd_cf(self, spot_at_cf_dates, forward_rates, hedge_ratio=1.0):

        # Dynamically adjust hedge ratio based on spot movement
        usd_cf = np.zeros((spot_at_cf_dates.shape[1], len(self.cash_flows_eur)))

        for index, (date, eur_cf) in enumerate(self.cash_flows_eur.items()):
            current_spot = spot_at_cf_dates[index, :]
            forward_rate = list(forward_rates.values())[index]

            moneyness = current_spot / forward_rate
            dynamic_ratio = np.clip(1.0 - 0.3 * (moneyness - 1.0), 0.5, 1.5)

            usd_cf[:, index] = eur_cf * (
                dynamic_ratio * forward_rate + (1 - dynamic_ratio) * current_spot
            )

        return usd_cf
