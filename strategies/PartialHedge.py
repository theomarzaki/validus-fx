from strategies.Hedging import HedgingStrategy
import numpy as np


class PartialForwardHedging(HedgingStrategy):
    def __init__(self, cash_flows_eur):
        super().__init__("Partial Hedge")
        self.cash_flows_eur = cash_flows_eur
        self.hedge_ratio=0.5

    def calculate_usd_cf(self, spot_at_cf_dates, forward_rates):
        usd_cf = np.zeros((spot_at_cf_dates.shape[1], len(self.cash_flows_eur)))

        for index, (date, eur_cf) in enumerate(self.cash_flows_eur.items()):
            forward_rate = list(forward_rates.values())[index]
            usd_cf[:, index] = eur_cf * (
                self.hedge_ratio * forward_rate
                + (1 - self.hedge_ratio) * spot_at_cf_dates[index, :]
            )

        return usd_cf
