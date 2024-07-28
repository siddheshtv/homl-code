import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(kind="scatter",
             x="GDP per capita (USD)",
             y="Life satisfaction",
             grid=True)
plt.axis([23_500, 62_500, 4, 9])
plt.savefig(fname="gdp_vs_life_sat.png", format="png")
plt.show()

model = LinearRegression()
kmodel = KNeighborsRegressor()

model.fit(X,y)
kmodel.fit(X,y)
X_new = [[37_655.2]]
print(model.predict(X_new))
print(kmodel.predict(X_new))