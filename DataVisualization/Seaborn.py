import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dat = sns.load_dataset("anscombe")

# sns.relplot(x="x", y="y", hue="dataset", data=dat)
# sns.catplot(x="x", y="y", hue="dataset", data=dat)

dat2 = sns.load_dataset("attention")

# sns.catplot(x="attention", y="score", kind="box", data=dat2)
# sns.catplot(x="attention", y="score", kind="boxen", data=dat2)

dat3 = sns.load_dataset("car_crashes")

# sns.relplot(x="alcohol", y="total", data=dat3)
# sns.relplot(x="alcohol", y="total", kind="line", data=dat3)
# sns.catplot(x="alcohol", y="total", data=dat3)

dat4 = sns.load_dataset("dots")
# sns.relplot(x="time", y="firing_rate", col="align", hue="choice",
#             kind="line", legend="full", data=dat4)

dat5 = pd.read_csv("student-mat.csv", sep=";")
# sns.relplot(x="G1", y="G3", col="school", hue="sex", style="sex", kind="line", data=dat5)

dat6 = sns.load_dataset("exercise")
# sns.catplot(x="id", y="pulse", col="diet", data=dat6)


dat7 = sns.load_dataset("flights")
# sns.catplot(x="month", y="passengers", hue="year", data=dat7)

dat8 = sns.load_dataset("penguins")
print(dat8.head())
sns.relplot(x="body_mass_g", y="culmen_length_mm", col="species", kind="line", hue="sex", data=dat8)


plt.show()
