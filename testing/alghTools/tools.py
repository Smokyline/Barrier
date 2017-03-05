import numpy as np
import pandas as pd
import math
import shutil
import os


def set_title_param(param):
    title = ''
    for key in ['q', 's', 'bar', 'delta', 'kmeans', 'alphaMax', 'pers', 'metrix', 'border']:
        value = param[key]
        if value is not False:
            title += '%s=%s ' % (key, value)
    return title


def res_to_txt(file, row):
    f = open(file, 'a')
    s = '{} {} {} {} {} {}'.format(row[0], row[1], row[2], row[3], row[4], row[5])  # name |B| acc s param
    f.write('%s\n' % s)
    f.close()


def points_diff_runnerAwB(A, B):
    """coord"""
    narrA = np.empty((0, 2))
    for i, a in enumerate(A):
        fDimEQLS = np.where(B[:, 0] == a[0])[0]
        if len(fDimEQLS) > 0:
            sDimEQLA = np.where(B[fDimEQLS, 1] == a[1])[0]
            if len(sDimEQLA) == 0:
                narrA = np.append(narrA, np.array([a]), axis=0)
        else:
            narrA = np.append(narrA, np.array([a]), axis=0)
    return narrA


def idx_diff_runnerAwB(A, B):
    '''idx'''
    narr = np.array([]).astype(int)
    for i, a in enumerate(A):
        if a not in B:
            narr = np.append(narr, a)
    return narr

def parseIdxH(lX, countX, h):
    finalIdx = []
    for i in range(lX):
        if countX[i] >= h:
            finalIdx.append(i)
    return np.array(finalIdx), h

def parseIdx_ro(lX, countX, r):
    """расчет границы по колмогоровском среднему"""
    not_zero_count = countX[np.where(countX > 0)]
    h = int((np.sum(not_zero_count ** r) / len(not_zero_count)) ** (1 / r))
    finalIdx = []
    for i in range(lX):
        if countX[i] >= h:
            finalIdx.append(i)
    return np.array(finalIdx), h



def read_cora_res(idxCX, c):
    CORAres = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/newEPA/Caucasus_CORA_result.csv',
                       ['idx', 'r1', 'r2', 'r3', 'r4']).T
    idxX = np.array([]).astype(int)
    for res in CORAres:
        if '+' in res[c]:
            idxRes = np.where(idxCX == res[0])[0][0]
            idxX = np.append(idxX, idxRes)
    return idxX


def create_folder(directory, title):
    q_dir = '%s%s%s' % (directory, title, os.path.sep)
    print('dir', q_dir)

    if not os.path.exists(q_dir):
        os.makedirs(q_dir)
        # os.makedirs('C:\\Users\\smoky\\Documents\\workspace\\result\\Barrier\\test \\')

    return q_dir


def read_csv(path, col):
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")
    for i, title in enumerate(col):
        try:
            array.append(frame[title].values)
        except:
            print('no_' + title, end=' ')

    return np.array(array)


def persRunner(X, pers, revers=False):
    print(X)
    border = int(len(X) * pers / 100)
    #if border >= len(X):
     #   border = len(X)-1
    sXV = np.argsort(X)
    if revers:
        return np.array(sXV[border:]).astype(int), sXV[border]

    else:
        return np.array(sXV[:border]).astype(int), sXV[border]



def find_VinXV(Xidx, idxXV, Vidx):
    w = []
    for i, x in enumerate(Xidx[idxXV]):
        if x in Vidx:
            w.append(i)
    return idxXV[w]


def km(data, k, randCZ=False):

    clusters = [[] for i in range(k)]
    idx_clusters = [[] for i in range(k)]

    centroids = np.array([])

    if randCZ:
        for cz_i in range(k):
            while True:
                cz = data[np.random.randint(len(data))]
                if cz not in centroids:
                    centroids = np.append(centroids, cz)
                    break
    else:
        unD = np.unique(data)
        for i in range(k):
            centroids = np.append(centroids, unD[i])

    def average(clusters):
        c_array = np.array([])
        for c in clusters:
            c_array = np.append(c_array, np.mean(c))
        return c_array

    itr = 1
    while True:
        for it, i in enumerate(data):
            evk_array = np.sqrt((i - centroids) ** 2)
            minIDX = np.argmin(evk_array)
            clusters[minIDX].append(i)
            idx_clusters[minIDX].append(it)

        oldCentroids = centroids.copy()
        centroids = average(np.array(clusters))

        if not np.array_equal(centroids, oldCentroids):
            clusters = [[] for i in range(k)]
            idx_clusters = [[] for i in range(k)]
        else:
            idx_clusters = np.array(idx_clusters)
            break
        itr += 1

    idxClusters_sort = [[] for i in range(k)]
    for i in range(len(centroids)):
        remove_index = np.argmin(centroids)
        #remove_index = np.argmax(centroids)

        idxClusters_sort[i] = idx_clusters[remove_index]
        idx_clusters = np.delete(idx_clusters, remove_index, 0)
        centroids = np.delete(centroids, remove_index)
    return np.array(idxClusters_sort), np.min(idxClusters_sort[-1])


def acc_check(result, EQ):
    accEQ = 0
    p1 = 25 / 111
    p2 = 50 / 111

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
