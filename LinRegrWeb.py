import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def sq_error(_x, _y, f_x=None):
    squared_error = []
    for i in range(len(_x)):
        squared_error.append((f_x(_x[i]) - _y[i])**2)
    return sum(squared_error)

data = pd.read_csv('web_traffic.tsv', sep='\t', header=None)

x, y = data[0], data[1]

lx = list(x)
ly = list(y)

for i in range(len(ly)):
    if math.isnan(ly[i]):
        ly[i] = 0
    else:
        ly[i] = ly[i]

x1 = [1, 743]

np_x = np.array(lx)
np_y = np.array(ly)

x2 = list(range(743))

th1_1, th0_1 = sp.polyfit(np_x, np_y, 1)
th2_2, th1_2, th0_2 = np.polyfit(np_x, np_y, 2)
th3_3, th2_3, th1_3, th0_3 = np.polyfit(np_x, np_y, 3)
th4_4, th3_4, th2_4, th1_4, th0_4 = np.polyfit(np_x, np_y, 4)
th5_5, th4_5, th3_5, th2_5, th1_5, th0_5 = np.polyfit(np_x, np_y, 5)

plt.scatter(lx, ly, label = u'Исходные данные', color='purple')

f1 = sp.poly1d(np.polyfit(np_x, np_y, 1))
plt.plot(x1, f1(x1), linewidth = 2, label = u'полином 1-ой степени')
f2 = sp.poly1d(np.polyfit(np_x, np_y, 2))
plt.plot(x2, f2(x2), linewidth = 2, label = u'полином 2-ой степени')
f3 = sp.poly1d(np.polyfit(np_x, np_y, 3))
plt.plot(x2, f3(x2), linewidth = 2, label = u'полином 3-ей степени')
f4 = sp.poly1d(np.polyfit(np_x, np_y, 4))
plt.plot(x2, f4(x2), linewidth = 2, label = u'полином 4-ой степени')
f5 = sp.poly1d(np.polyfit(np_x, np_y, 5))
plt.plot(x2, f5(x2), linewidth = 2, label = u'полином 5-ой степени')

fun1 = lambda x: th1_1*x + th0_1
fun2 = lambda x: th2_2*x**2 + th1_2*x + th0_2
fun3 = lambda x: th3_3*x**3 + th2_3*x**2 + th1_3*x + th0_3
fun4 = lambda x: th4_4*x**4 + th3_4*x**3 + th2_4*x**2 + th1_4*x + th0_4
fun5 = lambda x: th5_5*x**5 + th4_5*x**4 + th3_5*x**3 + th2_5*x**2 + th1_5*x + th0_5

res1 = sq_error(lx, ly, fun1)
res2 = sq_error(lx, ly, fun2)
res3 = sq_error(lx, ly, fun3)
res4 = sq_error(lx, ly, fun4)
res5 = sq_error(lx, ly, fun5)

print(f"Средняя квадратичная ошибка (1) составляет = {res1:.3f}")
print(f"Средняя квадратичная ошибка (2) составляет = {res2:.3f} на {100 - 100*res2/res1:.2f}% лучше (1)")
print(f"Средняя квадратичная ошибка (3) составляет = {res3:.3f} на {100 - 100*res3/res1:.2f}% лучше (1)")
print(f"Средняя квадратичная ошибка (4) составляет = {res4:.3f} на {100 - 100*res4/res1:.2f}% лучше (1)")
print(f"Средняя квадратичная ошибка (5) составляет = {res5:.3f} на {100 - 100*res5/res1:.2f}% лучше (1)")

from sklearn.linear_model import LinearRegression

model = LinearRegression()
np_x = np_x.reshape(-1, 1)
np_y = np_y.reshape(-1, 1)
model.fit(np_x, np_y)

x_predictions = np.array(list(range(744, 751)))
new_x_predictions = x_predictions.reshape(-1, 1)
y_predictions = model.predict(new_x_predictions)
y_predictions = y_predictions.flatten()

print("Предсказание значения целевого параметра для x = list(range(744, 751))")
new_th1_1, new_th0_1 = sp.polyfit(x_predictions, y_predictions, 1)
print(f"Полином 1-ой степени: {new_th0_1:.3f}x + {new_th1_1:.3f}")

new_th5_2, new_th4_2, new_th3_2, new_th2_2, new_th1_2, new_th0_2 = np.polyfit(x_predictions, y_predictions, 5)
print(f"Полином 5-ой степени: {new_th5_2:.6f}x^5 + {new_th4_2:.6f}x^4 + {new_th3_2:.6f}x^3 + {new_th2_2:.6f}x^2 + {new_th1_2:.6f}x + {new_th0_2:.6f}")

plt.title('Линейная регрессия')
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()
plt.savefig('plotweb.png')