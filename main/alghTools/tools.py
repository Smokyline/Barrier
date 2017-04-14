import numpy as np
import pandas as pd
import os
from main.alghTools.kmeans import km


def read_csv(path, col):
    """чтение csv файла по col колонкам"""
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")
    for i, title in enumerate(col):
        try:
            array.append(frame[title].values)
        except:
            print('no_' + title, end=' ')

    return np.array(array)


def read_cora_res(idxCX, c):
    """импорт результатов алгоритма EPA"""
    CORAres = read_csv('/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/kvz_CORA_result.csv',
                       ['idx', 'r1', 'r2', 'r3', 'r4']).T
    idxX = np.array([]).astype(int)
    for res in CORAres:
        if '+' in res[c]:
            idxRes = np.where(idxCX == res[0])[0][0]
            idxX = np.append(idxX, idxRes)
    return idxX


def set_title_param(param):
    """преобрахование значения перменных параметров в str """
    title = ''
    for key in ['s', 'vector', 'delta', 'kmeans', 'alphaMax', 'pers', 'metrics', 'mcos', 'nchCount', 'border', ]:
        value = param[key]
        if value is not False:
            title += '%s=%s ' % (key, value)
    return title


def res_to_txt(file, row):
    """сохранение результата алгоритма в txt файл """
    f = open(file, 'a')
    s = '{} {} {} {} {}'.format(row[0], row[1], row[2], row[3], row[4])  # name |B| acc s param
    f.write('%s\n' % s)
    f.close()


def save_xv_to_csv(X, i, folder, title):
    """сохранение расстояний XV в csv файл """
    path = '/Users/Ivan/Documents/workspace/result/Barrier/range/%s/' % folder
    if not os.path.exists(path):
        os.makedirs(path)
    XVdf = pd.DataFrame(np.array(X).ravel())
    name = '%s%s-%s' % (title, i[0], i[1])
    XVdf.to_csv(path + name + '.csv', index=False, header=False,
                sep=';', decimal=',')

def save_res_idx_to_csv(X, res, title):
    path = '/Users/Ivan/Documents/workspace/result/Barrier/csv_res/'
    if not os.path.exists(path):
        os.makedirs(path)
    one_zero_arr = []
    for i in range(len(X)):
        if i in res:
            one_zero_arr.append(1)
        else:
            one_zero_arr.append(0)

    XVdf = pd.DataFrame(np.array(one_zero_arr).ravel())
    XVdf.to_csv(path + title +'.csv', index=False, header=False,
                sep=';', decimal=',')


def h_separator(lX, countX, h):
    """Разделение множества попаданий по признаку по h границе"""
    finalIdx = []
    for i in range(lX):
        if countX[i] >= h:
            finalIdx.append(i)
    return np.array(finalIdx), h


def ro_separator(lX, countX, r):
    """Разделение множества попаданий по признаку по степенному среднему"""
    non_zero_count = countX[np.where(countX > 0)]
    h = (np.sum(non_zero_count ** r) / len(non_zero_count)) ** (1 / r)
    finalIdx = []
    for i in range(lX):
        if countX[i] >= h:
            finalIdx.append(i)
    return np.array(finalIdx), h


def pers_separator(X, pers, lower=False):
    """разделение множества по проценту от кол-ва"""
    border = int(len(X) * pers / 100)
    sXV = np.argsort(X).astype(int)
    #print('lenX', len(X))
    #print('uniqueX', len(np.unique(X)))
    #print('------------')
    if lower:
        out, const = sXV[:border], X[sXV[border]]
    else:
        out, const = sXV[len(sXV)-border:], X[sXV[len(sXV)-border]]
        #print('count', s)


    return out, const

def calc_count(full_XvF_count, roXvF, lX, nch=False, alpha_const_vF=None):
    """вычисление кол-ва попаданий X в вс класс по v малому f малому"""
    if not nch:
        full_XvF_count = np.ravel(full_XvF_count)
        countX = np.array([len(np.where(full_XvF_count == i)[0]) for i in range(lX)]).astype(int)
        return countX
    else:
        countX = np.zeros((1, lX))
        for i, Xvf in enumerate(roXvF):
            alpha_vf = alpha_const_vF[i]

            #nch_count_Xvf = np.array([alpha_vf / (max(alpha_vf, xvf)) for xvf in np.ravel(Xvf)])

            #Xvf = np.abs(1-np.ravel(Xvf))
            nch_count_Xvf = np.array([min(alpha_vf, xvf)/ alpha_vf for xvf in Xvf])

            countX += nch_count_Xvf
        return countX[0] / len(roXvF)


def count_border_blade(border, countX, border_const=None):
    """разделение кол-ва попаданий X по признакам в вс класс по v малому"""

    if border_const is None:
        '''если константа границы по кол-во попаданий в признаки не посчитана'''

        if border[0] == 'h(X)':
            idxXV, const = h_separator(len(countX), countX, h=border[1])
        elif border[0] == 'ro':
            idxXV, const = ro_separator(len(countX), countX, r=border[1])
        elif border[0] == 'pers':
            idxXV, const = pers_separator(countX, pers=border[1], lower=False)
        elif border[0] == 'kmeans':
            _, idxXV, const = km(countX, border[1], randCZ=False)
        else:
            print('wrong border')
            idxXV, const = None, None
        return np.array(idxXV).astype(int), const
    else:
        idxXV = np.array(np.where(countX >= border_const)[0]).astype(int)
        return idxXV

def found_nch_param_border(X, beta_p, mcos_p):
    if mcos_p is not False:
        beta = 0
        mcos = mcos_p
    else:
        beta = beta_p
        mcos = None

    def mq(X, s):
        mq_mean = np.mean(X ** s) ** (1 / s)
        return mq_mean

    def calc_scal_cos(v1, v2):
        return np.dot(v1, v2) / (np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)))

    def norm(X):
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    def found_mq_from_alpha(alpha, X):
        idxs = np.where(X <= alpha)[0]
        mq_idx = idxs[-1]
        return mq_idx

    mq_power = np.arange(0.5, 50, 0.1)

    #X = X[np.where(X > 0)]
    mq_value = [mq(X, s) for s in mq_power]
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

    if mcos_p is not False:
        border = mq_value[found_mq_from_alpha(mcos, vectors_cos)]
    else:
        alpha = calc_nch_alpha(vectors_cos, beta=beta)
        near_idx = found_mq_from_alpha(alpha, vectors_cos) - 1  # поиск наилучшей степени
        border = mq_value[near_idx]

    idx = np.where(X >= border)[0]

    # print('alpha', alpha)
    # print('mq_value', mq_value[near_alpha_drv])
    # print('mq_power', mq_power[near_alpha_drv])
    #print(len(idx))
    return idx, border

def calc_nch_alpha(X, beta):
    min_x = -0.998
    max_x = 1.
    epsl = 0.000000000000006
    X = X[np.where(X > 0)]

    def foo(B, a, beta):
        EPx = 0
        for b in B:
            #EPx += (b - a) / max(a, b)
            EPx += (a - b) / max(a, b)

        return EPx / len(B) - beta

    while True:
        half_x = (max_x + min_x) / 2
        fA_min = foo(X, min_x, beta)
        fA_max = foo(X, max_x, beta)
        fA_half = foo(X, half_x, beta)

        if fA_min == 0:
            alpha = min_x
            break
        if fA_half == 0:
            alpha = half_x
            break
        if fA_max == 0:
            alpha = max_x
            break

        if fA_min * fA_half < 0:
            max_x = half_x
        else:
            min_x = half_x

        if max_x - min_x < epsl:
            alpha = half_x
            break

    return alpha

def acc_check(result, EQ, grid=False):
    """вычисление точности алгоритма"""
    accEQ = 0
    if grid:
        r1, r2 = 15.5, 22
    else:
        r1, r2 = 25, 50
    p1 = r1 / 111
    p2 = r2 / 111

    for eq in EQ:
        if len(result) == 0:
            acc = 0
        else:
            evk = np.zeros((1, len(result)))
            for n, d in enumerate(eq):
                evk += (d - result[:, n]) ** 2
            evk = np.sqrt(evk[0])
            b_evk = evk[np.argmin(evk)]
            if b_evk <= p1:
                acc = 1
            elif p1 < b_evk <= p2:
                acc = (p2 - b_evk) / p1
            else:
                acc = 0
        accEQ += acc
    return round(accEQ / len(EQ), 4)


