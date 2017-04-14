from main.app.barrier_mod import BarrierMod, Result
from main.supportAlg.drawMap import Visual, check_pix_pers
from main.supportAlg.import_data import ImportData
from main.alghTools.tools import *
from testing.foo_anlz.foo_found import calc_a


import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_csv(path):
    """чтение csv файла по col колонкам"""
    frame = pd.read_csv(path, header=None, sep=';', decimal=",")
    array = np.array(frame.values)
    return array

imp = ImportData(gridVers=True)
imp.set_save_path('union_range')
vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
#vis.grid_res(imp.data_coord[res_idx], title=res.title, r=0.2)

COUNT_V = 17
COUNT_F = 3
param_name = ['H', 'B', 'M', 'A']

RO_V = [[]for v in range(COUNT_V)]

for param_n in param_name:
    path = '/Users/Ivan/Documents/workspace/result/Barrier/range/%srange/' % param_n
    for vi in range(COUNT_V):
        ro_Xv = read_csv(path + 'XVrange%i-%i.csv' % (vi + 1, 1)).T[0]
        ro_Xv = 1 - ro_Xv
        RO_V[vi].append(ro_Xv)

RO_V = np.array(RO_V)
lv, lf, lx = RO_V.shape

min_RO_X = np.zeros((lx, lv))
for i in range(lx):
    for j in range(lv):
        min_RO_X[i, j] = np.mean(RO_V[j, :, i])

RO_X = np.max(min_RO_X, axis=1)

#save_xv_to_csv(RO_X, [0, 0], 'union_range', 'ro_x')

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


def cos_parser(RO_X, mcos=0.9):
    """вектор степеней"""
    mq_power = np.arange(0.5, 100, 0.2)

    """импорт множества"""

    """вектор степенных средних"""
    mq_value = [mq(RO_X, s) for s in mq_power]
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

    # min_cos = np.min(vectors_cos[near_idx:])
    min_cos = mcos
    near_mcos_mq_idx = found_mq_from_alpha(min_cos, vectors_cos)
    near_mq_cos = mq_value[near_mcos_mq_idx]

    idx = np.where(RO_X >= near_mq_cos)[0]
    return idx

mcos = 0.88
full_idx = np.array([]).astype(int)
for vi in range(lv):
    ro_Xv = min_RO_X[:, vi]
    idx = cos_parser(ro_Xv, mcos)
    full_idx = np.append(full_idx, idx)

res_idx = np.unique(full_idx)
#res_idx = np.where(RO_X >= near_mq_cos)[0]
pers = check_pix_pers(imp.data_coord[res_idx], grid=imp.gridVers)  # процент занимаемой прощади
acc = acc_check(imp.data_coord[res_idx], imp.eq_all, grid=imp.gridVers)  # точность

#title = 'union_ro B=%i(%s) acc=%s mcos=%s' % (len(np.where(RO_X >= near_mq_cos)[0]), pers, acc, min_cos)
title = 'union_ro B=%i(%s) acc=%s mcos=%s' % (len(res_idx), pers, acc, mcos)
print(title)
vis.grid_res(imp.data_coord[res_idx], title=title, r=0.2)

save_res_idx_to_csv(imp.data_coord, res_idx, 'len=%s mcos=%s' % (len(res_idx), mcos))

"""
plt.subplot(211)
plt.plot(mq_power, mq_value, lw=1.5)
plt.plot(mq_power[vector_i_range], vectors_cos, lw=2, alpha=0.7)
plt.axvline(x=near_mq_power, ymin=0., ymax=0.99, lw=1, zorder=3, c='r', alpha=0.8)
plt.axvline(x=mq_power[near_mcos_mq_idx], ymin=0., ymax=0.99, lw=1, zorder=3, c='g', alpha=0.8)
plt.grid(True)
plt.title('beta=%s pow=%s mcos=%s' % (beta, mq_power[near_idx], min_cos))

plt.subplot(212)
#plt.plot(np.sort(data.ravel()))
plt.scatter(range(len(RO_X)), RO_X, s=3)
plt.plot([near_mq_value for i in range(len(RO_X))], color='r', lw=0.9)
plt.plot([near_mq_cos for i in range(len(RO_X))], color='g', lw=1.4, alpha=1)
plt.grid(True)


#plt.show()
"""
