class HedgingStrategy:
    
    def __init__(self, name):
        self.name = name

    def calculate_usd_cf(self,spot_at_cf_dates, forward_rates, hedge_ratio):
        return None