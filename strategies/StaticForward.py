from strategies.Hedging import HedgingStrategy
import numpy as np

class StaticForwardHedging(HedgingStrategy):
    def __init__(self,cash_flows_eur):
        super().__init__("Static Forward Hedge")
        self.cash_flows_eur = cash_flows_eur

    def calculate_usd_cf(self, spot_at_cf_dates, forward_rates, hedge_ratio=1.0):
        usd_cf = np.zeros((spot_at_cf_dates.shape[1],len(self.cash_flows_eur)))

        for index, (date, eur_cf) in enumerate(self.cash_flows_eur.items()):
            forward_rate = list(forward_rates.values())[index]
            usd_cf[:, index] = eur_cf * forward_rate

        return usd_cf