# coding=utf-8
import os

import numpy as np
import pandas as pd


def read_csv_pandas(path, header=False):
    df = pd.read_csv(path, delimiter=';', header=0, decimal=',')
    if header:
        header = df.columns.values.tolist()
        return np.array(df), np.array(header)
    else:
        return np.array(df)



def set_title_param(param):
    """преобрахование значения перменных параметров в str """
    title = ''
    for key in ['s', 'border', 'omega']:
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
    original_umask = os.umask(0)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    XVdf = pd.DataFrame(np.array(X).ravel())
    name = '%s%s-%s' % (title, i[0], i[1])
    XVdf.to_csv(path + name + '.csv', index=False, header=False,
                sep=';', decimal=',')
    os.umask(original_umask)


def save_res_idx_to_csv(one_zero_arr, title, path):
    path += '/csv_res/'
    original_umask = os.umask(0)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    XVdf = pd.DataFrame(np.array(one_zero_arr).ravel())
    XVdf.to_csv(path + title +'.csv', index=False, header=False,
                sep=';', decimal=',')
    os.umask(original_umask)

def save_res_coord_to_csv(coord, title, path):
    path += '/csv_res/'
    original_umask = os.umask(0)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    XVdf = pd.DataFrame(coord, columns=['x', 'y'])
    XVdf.to_csv(path + 'coord_' + title + '.csv', index=False, header=True,
                sep=';', decimal=',')
    os.umask(original_umask)

def coord_in_sample(xy, sample_coords):
    eps = 0.00000006
    ans = False

    x_dim = np.where(abs(sample_coords[:, 0] - xy[0]) < eps)[0]
    if len(x_dim) > 0:
        y_dim = np.where(abs(sample_coords[x_dim, 1] - xy[1]) < eps)[0]
        if len(y_dim) > 0:
                ans = True
    return ans


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
    non_zero_count = non_zero_count / 100
    h = (np.sum(non_zero_count ** r) / len(non_zero_count)) ** (1 / r)
    h = h*100
    finalIdx = []
    for i in range(lX):
        if countX[i] >= h:
            finalIdx.append(i)
    return np.array(finalIdx), h


def count_border_blade(border, countX):
    """разделение кол-ва попаданий X по признакам в вс класс по v малому"""

    if border[0] == 'h(X)':
        idxXV, const = h_separator(len(countX), countX, h=border[1])
    elif border[0] == 'ro':
        idxXV, const = ro_separator(len(countX), countX, r=border[1])
    else:
        print('wrong border')
        idxXV, const = None, None
    return np.array(idxXV).astype(int), const


def calc_count_X_in_F():
    pass


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



