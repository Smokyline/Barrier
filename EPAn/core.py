import numpy as np
from alghTools.tools import *
import time

def alphaM(X, V, FEAT, idxCX, idxCV, q, p, delta, bar):
    if bar:
        VV = barLength_2point(X, V, V, FEAT, p, delta)
    else:
        VV = length_2point(X, V, V, FEAT, p, delta)

    MqVV = mq_axis1(VV, q, alphaDem=False)
    alpha = max(MqVV)

    if bar:
        XV = barLength_2point(X, X, V, FEAT, p, delta)
    else:
        XV = length_2point(X, X, V, FEAT, p, delta)
    if q is None:
        MqXV = XV
    else:
        MqXV = mq_axis1(XV, q, alphaDem=False)

    idxXV = alphaParse(MqXV, alpha, kmeans=False, pers=False, alphaMax=alpha)
    idxVV = checkVinXV(idxCX, idxXV, idxCV)
    return idxXV, idxVV


def elsilon_module(X, V, FEAT, idxCX, idxCV, p, s, delta, kmeans, pers):
    sp = True
    VV = length_2point(X, V, V, FEAT, p, delta, sumP=sp)
    XV = length_2point(X, X, V, FEAT, p, delta, sumP=sp)

    epsX = calc_epsilon(XV, VV, FEAT, sumP=sp)

    idxXV = alphaParse(epsX, s, kmeans=kmeans, pers=pers)
    idxVV = checkVinXV(idxCX, idxXV, idxCV)
    #idxVV = epsX
    return idxXV, idxVV

def parseIdxH(lX, IDX, H, r, hcalc=None):
    h = H
    finalIdx = []

    if H is False:
        hC = hcalc[np.where(hcalc > 0)]
        h = int((np.sum(hC ** r) / len(hC)) ** (1 / r))

        print('h=%f' % h)

    for i in range(lX):
        if IDX[i] >= h:
            finalIdx.append(i)
    return np.array(finalIdx)


def mq_axis1(XV, q, alphaDem=False):
    l = len(XV[0])
    if alphaDem:
        for i in range(l):
            alpha = findAlpha(XV[:, i], s=1)
            XV[:, i] = XV[:, i] / alpha
        mq = (np.sum(XV ** q, axis=1) / l) ** (1 / q)
    else:
        mq = (np.sum(XV ** q, axis=1) / l) ** (1 / q)
    return mq

def alphaParse(XV, s, kmeans, pers, alphaMax=False):
    if type(pers) is int:
        X = np.ravel(XV)
        idx = persRunner(X, pers, revers=False)
        return idx
    elif kmeans is not False:
        idxXV = km(XV, kmeans, randCZ=False)[0]
        return np.array(idxXV).astype(int)
    else:
        if alphaMax is not False:
            alpha = alphaMax
        else:
            alpha = findAlpha(XV, s)
        idx = np.where(XV <= alpha)[0]
        return idx


def findAlpha(MqXV, s):
    MqXV = np.ravel(MqXV)
    if s is None:
        print('Error\nFalse alpha param  s is None')
        return None
    else:
        return (np.sum(MqXV ** s) / len(MqXV)) ** (1 / s)





def core(X, V, idxCX, idxCV, FEAT, q, p, s, bar=False, delta=False, kmeans=False, alphaDem=False,
         alphaMax=False, pers=False, epsilon=False):

    if alphaMax:
        idxXV, idxVV = alphaM(X, V, FEAT, idxCX, idxCV, q, p, delta, bar)
    elif epsilon:
        idxXV, idxVV = elsilon_module(X, V, FEAT, idxCX, idxCV, p, s, delta, kmeans, pers)

    else:
        if bar:
            XV = barLength_2point(X, X, V, FEAT, p, delta)
        else:
            #time_start = int(round(time.time() * 1000))
            XV = length_2point(X, X, V, FEAT, p, delta)
            #print(int(round(time.time() * 1000)) - time_start)

        if len(XV[0]) > 1:
            if q is None:
                print('Error\nq is None & v!=1')
            else:
                XV = mq_axis1(XV, q, alphaDem)

        idxXV = alphaParse(XV, s, kmeans=kmeans, pers=pers)
        idxVV = checkVinXV(idxCX, idxXV, idxCV)

    return idxXV, idxVV
