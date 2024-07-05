import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

print(data)

"""def loss_function(w, b, pontos):
    total_error = 0
    for i in range(len(pontos)):
        x = pontos.iloc[i].Hours
        y = pontos.iloc[i].Scores

        total_error = total_error + (((w * x + b) - y) ** 2)
    total_error = total_error / len(pontos)
    
    return total_error"""

def gradient_descent(w_now, b_now, pontos, L):
    w_gradient = 0
    b_gradient = 0

    n = len(pontos)

    for i in range(n):
        x = pontos.iloc[i].Hours
        y = pontos.iloc[i].Scores

        w_gradient = w_gradient + -(2/n) * x * (y - (w_now * x + b_now))
        b_gradient = b_gradient + -(2/n) * (y - (w_now * x + b_now))

    m = w_now - w_gradient * L
    b = b_now - b_gradient * L

    return m, b 

w = 0
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print("iteração: ", i)
    w, b = gradient_descent(w, b, data, L)

print(w, b)

plt.scatter(data.Hours, data.Scores, color="black")
plt.plot(list(range(0,10)), [w * x + b for x in range(0, 10)], color="red")
plt.show()