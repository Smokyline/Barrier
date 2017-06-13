import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from test_f.foo_anlz.foo_found import calc_a

def read_csv(path):
    """чтение csv файла по col колонкам"""
    frame = pd.read_csv(path, header=None, sep=';', decimal=",")
    array = np.array(frame.values)
    return array

def mq(X, s):
    mq_mean = np.mean(X**s) ** (1/s)
    return mq_mean

def norm(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

"""вектор степеней"""
mq_power = np.arange(-50, 0, 1)
#mq_power = np.arange(0, 51, 1)

"""импорт множества"""
path = '/Users/Ivan/Documents/workspace/result/Barrier/XVrange/XV_11.csv'
#path = '/Users/Ivan/Documents/workspace/result/Barrier/count/count_1.csv'
data = read_csv(path)

"""вектоп степенный средних"""
mq_value = [mq(data, s) for s in mq_power]

deriv = np.array([mq_value[i+1] - mq_value[i] for i in range(len(mq_value)-1)])
#deriv2 = np.array([deriv[i+2] - (2*deriv[i+1]) + deriv[i] for i in range(len(deriv)-2)])
#deriv3 = np.array([deriv2[i+3] - (3*deriv2[i+2]) + (3*deriv[i+1]) - deriv2[i] for i in range(len(deriv2)-3)])
#deriv3 = deriv2

final_deriv = norm(deriv)


alpha = calc_a(final_deriv, beta=0.5)
near_alpha_drv = np.argmin(np.abs(final_deriv - alpha))  # ближайшая степень к alpha

print('alpha', alpha)
print('mq_value', mq_value[near_alpha_drv])
print('mq_power', mq_power[near_alpha_drv])

plt.figure(1)
plt.subplot(211)

plt.plot(mq_power, mq_value, lw=1.5)
plt.plot(mq_power[:-1], final_deriv, lw=2, alpha=0.7)
plt.axvline(x=mq_power[near_alpha_drv], ymin=0., ymax=0.99, lw=1.3, zorder=3, c='r')
plt.grid(True)

#plt.subplot(212)
#plt.plot(np.sort(data.ravel()))
#plt.plot([mq_value[near_alpha_drv] for i in range(len(data))], color='r')
#plt.grid(True)


plt.show()
