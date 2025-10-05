import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x =np.array ([[1], [2], [3], [4], [5]])
y= np.array([2,4,5,4,2])

model= LinearRegression()
model.fit(x,y)

y_pred = model.predict(x)

plt.scatter(x,y, color="red")
plt.plot(x,y_pred, color="blue")

plt.show()