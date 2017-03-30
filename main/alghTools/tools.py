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
    for key in ['q', 's', 'vector', 'delta', 'kmeans', 'alphaMax', 'pers', 'metrix', 'nchCount', 'border', ]:
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


def save_xv_to_csv(XV, vi):
    """сохранение расстояний XV в csv файл """
    path = '/Users/Ivan/Documents/workspace/result/Barrier/XVrange/'
    if not os.path.exists(path):
        os.makedirs(path)
    XVdf = pd.DataFrame(np.abs(np.array(XV).ravel() - 1))
    XVdf.to_csv(path + 'XV_' + str(vi + 1) + '.csv', index=False, header=False,
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


def pers_separator(X, pers, revers=False):
    """разделение множества по проценту от кол-ва"""
    border = int(len(X) * pers / 100)
    sXV = np.argsort(X)
    if revers:
        """если revers=True последние pers """
        return np.array(sXV[border:]).astype(int), sXV[border]

    else:
        """если revers=False первые pers """
        return np.array(sXV[:border]).astype(int), sXV[border]


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
            nch_count_Xvf = np.array([alpha_vf / (max(alpha_vf, xvf)) for xvf in np.ravel(Xvf)])
            countX += nch_count_Xvf
        return countX[0]


def count_border_blade(border, countX, border_const=None):
    """разделение кол-ва попаданий X по признакам в вс класс по v малому"""

    if border_const is None:
        '''если константа границы по кол-во попаданий в признаки не посчитана'''

        if border[0] == 'h(X)':
            idxXV, const = h_separator(len(countX), countX, h=border[1])
        elif border[0] == 'ro':
            idxXV, const = ro_separator(len(countX), countX, r=border[1])
        elif border[0] == 'pers':
            idxXV, const = pers_separator(countX, pers=border[1], revers=True)
        elif border[0] == 'kmeans':
            _, idxXV, const = km(countX, border[1], randCZ=False)

        else:
            print('wrong border')
            idxXV, const = None, None

        return np.array(idxXV).astype(int), const

    else:
        idxXV = np.array(np.where(countX >= border_const)[0]).astype(int)
        return idxXV



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

