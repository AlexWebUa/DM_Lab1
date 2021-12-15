import pandas as pd
from matplotlib import pyplot as plt

sample = pd.read_csv("countries.csv")
us = sample[sample.country == "United States"]
china = sample[sample.country == "China"]
plt.plot(us.year, us.population / 10**6)
plt.plot(china.year, china.population / 10**6)
plt.xlabel("year")
plt.ylabel("population, mln")
plt.legend(["us", "china"])
plt.show()
