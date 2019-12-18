import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

lr=LinearRegression()

df = pd.read_csv("polynominalReg.csv",sep=";")

x=df.fiyat.values.reshape(-1,1)
y=df.hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("Fiyat")
plt.ylabel("Hiz")

lr.fit(x,y)
yhead=lr.predict(x)
plt.plot(x,yhead,color="red",label="linear")

pr = PolynomialFeatures(degree=2)
xp = pr.fit_transform(x)

lr2= LinearRegression()
lr2.fit(xp,y)
yhead2 = lr2.predict(xp)

plt.plot(x,yhead2,color="green",label="poly(degree=2)")

pr2 = PolynomialFeatures(degree=4)
xp=pr2.fit_transform(x)

lr2.fit(xp,y)
yhead3 = lr2.predict(xp)

plt.plot(x,yhead3,color="purple",label="poly(degree=4)")

plt.legend()
plt.show()

