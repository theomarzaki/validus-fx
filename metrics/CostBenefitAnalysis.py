import pandas as pd


def calculate_cost_benefit_analysis(results):

    no_hedge = results.iloc[0]
    hedging_strategies = results.iloc[0:]

    cba = []
    for index, strategy in hedging_strategies.iterrows():
        irr_preserve = no_hedge["IRR"].mean() - strategy["IRR"].mean()
        risk_reduction = (no_hedge["IRR"].std() - strategy["IRR"].std()) / no_hedge[
            "IRR"
        ].std()
        var_reduction = (no_hedge["VaR"] - strategy["VaR"]) / no_hedge["VaR"]

        cba.append(
            {
                "Strategy Name": strategy["Strategy Name"],
                "IRR Preservation": irr_preserve,
                "Risk Reduction": risk_reduction,
                "VaR Reduction": var_reduction,
                "Weighted Analysis": 0.33 * irr_preserve
                + 0.33 * risk_reduction
                + 0.33 * var_reduction,
            }
        )

    return pd.DataFrame(cba)
