from simple_linear_regresion import SimpleLinearRegresion
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('advertising.csv', delimiter=',')
x = data['TV']
y = data['Sales']

model = SimpleLinearRegresion(x, y)

y_pred = model.predict()
print(y_pred)

accuaracy = model.r_square()
print(accuaracy)

plt.scatter(x, y)
plt.plot(x, y_pred, 'r')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()
