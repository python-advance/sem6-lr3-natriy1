import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

def gradient(x, y, alpha, n):
    m = len(x)
    theta_0, theta_1 = 0, 0
    for i in range(n):
        sum_1 = 0
        for i in range(m):
            sum_1 += theta_0 + theta_1 * x[i] - y[i]
        temp1 = theta_0 - alpha * (1 / m) * sum_1
        sum_2 = 0
        for i in range(m):
            sum_2 += (theta_0 + theta_1 * x[i] - y[i]) * x[i]
        temp2 = theta_1 - alpha * (1 / m) * sum_2
        theta_0, theta_1 = temp1, temp2
    return theta_0, theta_1


data = pd.read_csv('ex1data1.csv', header=None)
x, y = data[0], data[1]

x1 = [1, 25]
y1 = [0, 0]
th0, th1 = gradient(x, y, 0.001, len(x))
y1[0] = th0 + x1[0] * th1
y1[1] = th0 + x1[1] * th1

plt.plot(x1, y1, 'purple', label = u'Вручную')
plt.scatter(x, y, label = u'Ex1data1.csv', color='b')

np_x = np.array(x)
np_y = np.array(y)
new_th1, new_th0 = (np.polyfit(np_x, np_y, 1)).tolist()

new_y1 = [0, 0]
new_y1[0] = new_th0 + x1[0] * new_th1
new_y1[1] = new_th0 + x1[1] * new_th1
plt.plot(x1, new_y1, 'red', label = u'С Polyfit')

print(new_th0, new_th1)

plt.title('Линейная регрессия с одной переменной. Градиентный спуск')
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()
plt.savefig('plot1.png')