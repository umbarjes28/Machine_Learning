from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[: -20]

diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[: -20]
diabetes_Y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_Y_train)


print('Input Values')
print(diabetes_X_test)

diabetes_Y_pred = regr.predict(diabetes_X_test)

print("Predicted output values")
print(diabetes_Y_pred)

plt.scatter(diabetes_X_test, diabetes_Y_test, color='black')
plt.plot(diabetes_X_test, diabetes_Y_test, color='red', linewidth=1)

plt.show()
