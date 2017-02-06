import numpy as np


def findF(X, maxF, minF):
    Fx = 0
    for x in X:
        if minF <= x <= maxF:
            Fx += 1
    return Fx

def length_2point(XX, X, V, F, p):
    XV = np.empty((len(F), len(X), len(V)))
    lengthAx = len(XX)
    for f in range(len(F)):
        for iX, x in enumerate(X[:, f]):
            for iV, v in enumerate(V[:, f]):
                minF = min(x, v)
                maxF = max(x, v)
                Fx = findF(XX[:, f], maxF, minF) / lengthAx
                XV[f, iX, iV] = Fx
    pFEAT = (np.sum(XV ** p, axis=0) / len(F)) ** (1 / p)
    return pFEAT

def findAlpha(MqXV, MqVV, s):
    if s is None:
        return max(MqVV)
    else:
        return (np.sum(MqXV ** s) / len(MqXV)) ** (1 / s)

def alphaParse(XV, alpha):
    idx = np.array([])
    for i, xv in enumerate(XV):
        if xv <= alpha:
            idx = np.append(idx, i)
    return idx.astype(int)


def parseIdx(lX, IDX, H, r):
    h = H
    finalIdx = []
    hC = []
    if H is None:
        for i in range(lX):
            countIdx = len(np.where(IDX == i)[0])
            hC.append(countIdx)
        hC = np.array(hC)
        hcIDX = np.where(hC > 0)[0]
        hC = hC[hcIDX]
        h = (np.sum(hC ** r) / len(hC)) ** (1 / r)
    for i in range(lX):
        countIdx = len(np.where(IDX == i)[0])
        if countIdx >= h:
            finalIdx.append(i)
    return np.array(finalIdx)


def checkVinXV(Xidx, idxXV, Vidx):
    w = []
    for i, x in enumerate(Xidx[idxXV]):
        if x in Vidx:
            w.append(i)
    return idxXV[w]

def mQ(X, q):
    l = len(X[0])
    mq = (np.sum(X ** q, axis=1) / l) ** (1 / q)
    return mq

def coreS(X, V, idxCX, idxCV, FEAT, q, p, s):


    return idxXV, idxVV
