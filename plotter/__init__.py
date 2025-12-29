import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (30, 20)})
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["grid.color"] = "#cccccc"

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=["#009392", "#6969AA", "#E88471", "#E9E29C", "#EEB479" ]
)

sns.set_context(
    "paper", font_scale=4, rc={"lines.linewidth": 5}
)