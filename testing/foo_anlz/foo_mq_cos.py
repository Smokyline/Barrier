import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from testing.foo_anlz.foo_found import calc_a

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

def calc_scal_cos(v1, v2):
    #return (v1[0]*v2[0] + v1[1]*v2[1]) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
    #return 1+(v1[1]*v2[1]) / (np.sqrt(1 + v1[1]**2) * np.sqrt(1+ v2[1]**2))

def found_mq_from_alpha(alpha, X):
    idxs = np.where(X <= alpha)[0]
    mq_idx = idxs[-1]
    return mq_idx

def psi_foo(roXvF, alphaF):
    lX = len(roXvF[0])
    countX = np.zeros((1, lX))
    for i, Xvf in enumerate(roXvF):
        alpha_vf = alphaF[i]
        nch_count_Xvf = np.array([min(alpha_vf, xvf) / alpha_vf for xvf in Xvf])
        countX += nch_count_Xvf
    return countX[0] / len(roXvF)



"""вектор степеней"""
#mq_power = np.arange(-50, 0, 0.2)
mq_power = np.arange(0.5, 100, 0.2)


"""импорт множества"""
path = '/Users/Ivan/Documents/workspace/result/Barrier/range/B3range/XVrange%i-%i.csv' % (1, 0)
#path = '/Users/Ivan/Documents/workspace/result/Barrier/count/count%i-%i.csv' % (0, 0)
data = read_csv(path).T[0]
data = np.abs(data - 1)

print('data:', data)
print('max min data', max(data), min(data))

"""вектор степенных средних"""
mq_value = [mq(data, s) for s in mq_power]
f = mq_value

vectors_cos = []
vector_i_range = np.arange(1, len(mq_value) - 1)
for i in vector_i_range:
    v1 = np.array([-1, f[i - 1] - f[i]])
    v2 = np.array([1, f[i + 1] - f[i]])
    cos_v1v2 = calc_scal_cos(v1, v2)
    vectors_cos.append(cos_v1v2)

vectors_cos = np.abs(np.array(vectors_cos))
vectors_cos = norm(vectors_cos)

beta = 0
alpha = calc_a(vectors_cos, beta)
print('alpha', alpha)
print('max min cos', max(vectors_cos), min(vectors_cos))
near_idx = found_mq_from_alpha(alpha, vectors_cos) - 1  # поиск наилучшей степени
near_mq_value = mq_value[near_idx]
near_mq_power = mq_power[near_idx]

# min_cos = np.min(vectors_cos[near_idx:])
min_cos = 0.9999
near_mcos_mq_idx = found_mq_from_alpha(min_cos, vectors_cos)
#near_mq_cos = mq_value[near_mcos_mq_idx]
near_mq_cos = np.min(data[np.where(data >= 0.95)])

print('count', len(np.where(data > near_mq_value)[0]))
print('mq_value', near_mq_value)
print('mq_power', mq_power[near_idx])
print('min cos b=0', min_cos)
#print('len Xmcos', len(np.where(data >= near_mq_cos)[0]))



#f = open('/Users/Ivan/Documents/workspace/result/Barrier/XVrange/mincos.txt', 'a')
#f.write('%s\n' % np.min(vectors_cos[near_idx:]))
#f.close()

plt.subplot(211)
plt.plot(mq_power, mq_value, lw=1.5)
plt.plot(mq_power[vector_i_range], vectors_cos, lw=2, alpha=0.7)
plt.axvline(x=near_mq_power, ymin=0., ymax=0.99, lw=1, zorder=3, c='r', alpha=0.8)
plt.axvline(x=mq_power[near_mcos_mq_idx], ymin=0., ymax=0.99, lw=1, zorder=3, c='g', alpha=0.8)
plt.grid(True)
plt.title('beta=%s pow=%s mcos=%s' % (beta, mq_power[near_idx], min_cos))

plt.subplot(212)
#plt.plot(np.sort(data.ravel()))
plt.scatter(range(len(data)), data, s=3)
plt.plot([near_mq_value for i in range(len(data))], color='r', lw=0.9)
plt.plot([near_mq_cos for i in range(len(data))], color='g', lw=1.4, alpha=1)
plt.grid(True)


plt.show()

