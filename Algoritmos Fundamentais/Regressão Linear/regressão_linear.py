import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

print(data)

def loss_function(w, b, pontos):
    total_error = 0
    for i in range(len(pontos)):
        x = pontos.iloc[i].Hours
        y = pontos.iloc[i].Scores

        total_error = total_error + (((w * x + b) - y) ** 2)
    total_error = total_error / len(pontos)
    
    return total_error

def gradient_descent(w_now, b_now, pontos, Learning_rate):
    w_gradient = 0
    b_gradient = 0

    n = len(pontos)

    for i in range(n):
        x = pontos.iloc[i].Hours
        y = pontos.iloc[i].Scores

        w_gradient = w_gradient + -(2/n) * x * (y - (w_now * x + b_now))
        b_gradient = b_gradient + -(2/n) * (y - (w_now * x + b_now))

    w = w_now - w_gradient * Learning_rate
    b = b_now - b_gradient * Learning_rate

    return w, b 

def train(data, Learning_rate, epochs):
    w = 0
    b = 0

    for i in range(epochs):
        w, b = gradient_descent(w, b, data, Learning_rate)
        if i % 50 == 0:
            print("iteração: ", i, "perda: ", loss_function(w, b, data))

    return w, b

def predict(x, w, b):
    return w*x + b

w, b = train(data, 0.0001, 1000)
x_new = 8.5
y_new = predict(x_new, w, b)
print(y_new)

"""data_predict = pd.read_csv('data_predict.csv')
for i in range(len(data_predict)):
    data_predict.iloc[i].Scores = predict(data_predict.iloc[i].Hours, w, b)
    print(data_predict.iloc[i].Hours, data_predict.iloc[i].Scores)
#print(data_predict)    """

plt.scatter(data.Hours, data.Scores, color="black")
plt.scatter(x_new, y_new, color="green")
plt.plot(list(range(0,10)), [w * x + b for x in range(0, 10)], color="red")
plt.show()