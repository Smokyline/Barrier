import numpy as np
import pandas as pd
import math
import shutil
import os




def eq_os_import():
    file_name_all = 'kvz_eq6_all.csv'
    file_name_ist = 'kvz_eq6_istor.csv'
    file_name_inst = 'kvz_eq6_instr.csv'
    legend = ['M6+', 'M6+ istor', 'M6+ instr']
    if os.name == 'nt':
        return read_csv('C:\\Users\\smoky\\Documents\\workspace\\resourses\\csv\\geop\\kvz\\'+file_name_all, ['x', 'y']).T
    if os.name == 'posix':
        eq_all = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/geop/kvz/'+file_name_all, ['x', 'y']).T
        eq_ist = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/geop/kvz/'+file_name_ist, ['x', 'y']).T
        eq_inst = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/geop/kvz/'+file_name_inst, ['x', 'y']).T
        return eq_all, eq_ist, eq_inst, legend

def set_title_param(param):
    title = ''
    for key in ['q', 's', 'r', 'bar', 'delta', 'kmeans', 'alphaDem', 'alphaMax', 'pers', 'epsilon']:
        value = param[key]
        if value is not False:
            title += '%s=%s ' % (key, value)
    return title

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

def read_cora_res(idxCX, c):
    CORAres = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/newEPA/Caucasus_CORA_result.csv',
                       ['idx', 'r1', 'r2', 'r3', 'r4']).T
    idxX = np.array([]).astype(int)
    for res in CORAres:
        if '+' in res[c]:
            idxRes = np.where(idxCX == res[0])[0][0]
            idxX = np.append(idxX, idxRes)
    return idxX

def findF(X, lX, maxF, minF):
    range_x = np.where(np.logical_and(X >= minF, X <= maxF))[0] #7.3
    #range_x = np.where((X >= minF) & (X <= maxF))[0] #7.5
    return len(range_x) / lX

def findFdelta(X, maxF, minF):
    delta = 0
    for x in X:
        if minF <= x <= maxF:
            pass
        else:
            piX1 = abs(x - minF)
            piX2 = abs(x - maxF)
            minPix = min(piX1, piX2)
            maxPix = max(piX1, piX2)
            delta += (minPix / maxPix)
    return delta

def length_2point(XX, X, V, F, p, delta=False, sumP=True):
    XV = np.zeros((len(X), len(V)))
    lengthAx = len(XX)
    for f in range(len(F)):
        XXf = XX[:, f]
        for iX, x in enumerate(X[:, f]):
            xi_array = [findF(XXf, lengthAx, max(x, v), min(x, v)) for v in V[:, f]]
            XV[iX] += xi_array
    return XV

def findFbar(X, x, v, F):
    Fx = 0
    for xx in X:
        for f in range(len(F)):
            maxF = max(x[f], v[f])
            minF = min(x[f], v[f])
            if minF <= xx[f] <= maxF:
                Fx += 1
    return Fx / len(F)

def findFbarDelta(X, x, v, F):
    Fx = 0
    for xx in X:
        for f in range(len(F)):
            maxF = max(x[f], v[f])
            minF = min(x[f], v[f])
            if minF <= xx[f] <= maxF:
                pass
            else:
                piX1 = abs(xx[f] - minF)
                piX2 = abs(xx[f] - maxF)
                minPix = min(piX1, piX2)
                maxPix = max(piX1, piX2)
                Fx += (minPix / maxPix)
    return Fx / len(F)

def barLength_2point(XX, X, V, F, p, delta=False):
    XV = np.empty((len(F), len(X), len(V)))
    lengthAx = len(XX)
    for iF, vectF in enumerate(F):
        for iX, x in enumerate(X[:, vectF]):
            for iV, v in enumerate(V[:, vectF]):
                if delta:
                    Fx = 1 - (findFbarDelta(XX, x, v, vectF) / lengthAx)
                else:
                    Fx = findFbar(XX, x, v, vectF) / lengthAx
                XV[iF, iX, iV] = Fx
    pFEAT = (np.sum(XV ** p, axis=0) / len(F)) ** (1 / p)
    return pFEAT



def calc_epsilon(XV, VV, FEAT, sumP):
    if sumP is False:
        epsXP = np.empty((0, len(XV[0])))
        for f in range(len(FEAT)):
            epsXpArray = np.array([])
            for x in XV[f]:
                epsXV = 0
                for v in VV[f]:
                    epsXV += np.sum(np.power(x - v, 2))
                    # epsXV += np.sum(abs(x - v))
                epsXpArray = np.append(epsXpArray, epsXV)
            epsXP = np.append(epsXP, [epsXpArray], axis=0)
        epsXP = np.mean(epsXP, axis=0)

    else:
        epsXP = np.array([])
        for x in XV:
            epsXV = 0
            for v in VV:
                epsXV += np.sum(np.power(x - v, 2))
                # epsXV += np.sum(abs(x - v))
            epsXP = np.append(epsXP, epsXV)

    return epsXP

def create_folder(directory, title):
    q_dir = '%s%s%s' % (directory, title, os.path.sep)
    print('dir', q_dir)

    if not os.path.exists(q_dir):
        os.makedirs(q_dir)
        #os.makedirs('C:\\Users\\smoky\\Documents\\workspace\\result\\Barrier\\test \\')

    return q_dir

def read_csv(path, col=['idx', 'Hmax', 'Hmin', 'DH', 'Top', 'Q', 'HR', 'Nl',
                        'Rint', 'DH/l', 'Nlc', 'R1', 'R2', 'Bmax', 'Bmin', 'DB',
                        'Mmax', 'Mmin', 'DM', 'dps', 'Hdisp', 'Bdisp']):
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")
    for i, title in enumerate(col):
        try:
            array.append(frame[title].values)
        except:
            print('no_' + title, end=' ')

    return np.array(array)

def persRunner(X, pers, revers=False):
    border = int(len(X) * pers / 100)
    if revers:
        sXV = np.argsort(X)[::-1]
    else:
        sXV = np.argsort(X)

    return sXV[:border]

def find_psi(idxXV, idxVV, CX):
    idxwV = np.array([]).astype(int)
    for i, x in enumerate(idxXV):
        if x not in idxVV:
            idxwV = np.append(idxwV, i)
    return idxwV, max(CX[idxwV])


def psiX(psi, CX, idxFX, idxwW):
    idxXwV = idxFX[idxwW]
    CX = CX[idxwW]

    idxNewVx = np.array([]).astype(int)
    for i, idxx in enumerate(idxXwV):
        if CX[i] >= psi:
            idxNewVx = np.append(idxNewVx, idxx)

    return idxXwV.astype(int), idxNewVx.astype(int)



def checkVinXV(Xidx, idxXV, Vidx):
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

        idxClusters_sort[i] = idx_clusters[remove_index]
        idx_clusters = np.delete(idx_clusters, remove_index, 0)

        centroids = np.delete(centroids, remove_index)
    return np.array(idxClusters_sort)


def acc_check(result):
    EQ, _, _, _ = eq_os_import()
    accEQ = 0
    p1 = 25 / 111
    p2 = 50 / 111

    for eq in EQ:
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
    return round(accEQ / len(EQ), 2)

