from sklearn.linear_model import Perceptron
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

print("AND Perceptron:")
andPerceptron = Perceptron(max_iter=10, eta0=0.5, random_state=0)
andPerceptron.fit(X, y_and)

print("Pesos:", andPerceptron.coef_)
print("Bias:", andPerceptron.intercept_)

for i, x in enumerate(X):
    pred = andPerceptron.predict([x])[0]
    print(f"Entrada: {x} \tPredicción: {pred} \tEsperado: {y_and[i]}")

print("\n" + "-"*40 + "\n")

print("OR Perceptron:")
orPerceptron = Perceptron(max_iter=10, eta0=0.5, random_state=0)
orPerceptron.fit(X, y_or)

print("Pesos:", orPerceptron.coef_)
print("Bias:", orPerceptron.intercept_)

for i, x in enumerate(X):
    pred = orPerceptron.predict([x])[0]
    print(f"Entrada: {x} \tPredicción: {pred} \tEsperado: {y_or[i]}")

