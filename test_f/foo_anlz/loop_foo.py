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


def calc_scal_cos(v1, v2):
    #return (v1[0]*v2[0] + v1[1]*v2[1]) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
    #return 1+(v1[1]*v2[1]) / (np.sqrt(1 + v1[1]**2) * np.sqrt(1+ v2[1]**2))

def found_mq_from_alpha(alpha, X, count=False):
    idxs = np.where(X <= alpha)[0]
    if count:
        mq_idx = idxs[-1]
    else:
        mq_idx = idxs[0]
    return mq_idx

"""вектор степеней"""
#mq_power = np.arange(-75, 0, 0.2)
mq_power = np.arange(1, 74, 0.5)

MQ_VALUE = []
MQ_VALUE_NEAR = []
COS = []
MQ_POWER = []
DATA = []

for j in np.arange(1, 18, 1):
    """импорт множества"""
    #path = '/Users/Ivan/Documents/workspace/result/Barrier/XVrange/XVrange_%s.csv' % j
    path = '/Users/Ivan/Documents/workspace/result/Barrier/count/count_%s.csv' % j
    data = read_csv(path)
    count = True

    """вектор степенных средних"""
    mq_value = [mq(data, s) for s in mq_power]
    f = mq_value

    vectors_cos = []
    vector_i_range = np.arange(1, len(mq_value) - 1)
    for i in vector_i_range:
        # v1 = np.array([mq_power[i-1], f[i-1] - f[i]])
        # v2 = np.array([mq_power[i+1], f[i+1] - f[i]])
        v1 = np.array([-1, f[i - 1] - f[i]])
        v2 = np.array([1, f[i + 1] - f[i]])
        cos_v1v2 = calc_scal_cos(v1, v2)
        vectors_cos.append(cos_v1v2)
    vectors_cos = np.abs(np.array(vectors_cos))
    vectors_cos = norm(vectors_cos)

    beta = 0
    alpha = calc_a(vectors_cos, beta)
    near_alpha_drv = found_mq_from_alpha(alpha, vectors_cos, count=count)  # ближайшая степень к alpha
    near_mq = mq_power[near_alpha_drv]

    #print('alpha', alpha)
    #print('mq_value', mq_value[near_alpha_drv])
    #print('mq_power', mq_power[near_alpha_drv])

    MQ_VALUE.append(mq_value)
    MQ_VALUE_NEAR.append(mq_value[near_alpha_drv])
    MQ_POWER.append(mq_power[near_alpha_drv])
    COS.append(vectors_cos)
    DATA.append(data)



fig, axs = plt.subplots(9,2, figsize=(15, 6), facecolor='w', edgecolor='k')
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
axs = axs.ravel()
#fig.subplots_adjust(hspace = .5, wspace=.001)


for k in range(len(COS)):
    mq_value = MQ_VALUE[k]
    deriv = COS[k]
    near_mq_power = MQ_POWER[k]
    data = DATA[k]
    near_mq_value = MQ_VALUE_NEAR[k]

    #wh = len(np.where(data>near_mq_value)[0])
    #print('v%s c:%s' % (k+1, wh))

    ax = axs[k]

    ax.plot(mq_power, mq_value, lw=1.5)
    ax.plot(mq_power[1:-1], deriv, lw=2, alpha=0.7)
    ax.axvline(x=near_mq_power, ymin=0., ymax=0.99, lw=1.3, zorder=3, c='r')

    #ax.scatter(range(len(data)), data, s=3)
    #ax.plot([near_mq_value for i in range(len(data))], color='r')
    #ax.grid(True)

    ax.grid(True)
    ax.set_ylabel('V_%s' % (k+1), fontsize=5)

plt.savefig('/Users/Ivan/Documents/workspace/result/Barrier/XVrange/nchcountgraf_beta0.png', dpi=500)
#plt.savefig('/Users/Ivan/Documents/workspace/result/Barrier/XVrange/nchcount_beta0.png', dpi=500)

#plt.show()
