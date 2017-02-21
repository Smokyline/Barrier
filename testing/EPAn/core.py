import numpy as np
from testing.alghTools.tools import *
import time
import math



def mq_axis1(XV, q):
    """степенное среднее каждой строки"""
    ax1 = len(XV[0])
    if q is None:
        print('Error\nq is None or v!=1')
        mq_array = None
    else:
        mq_array = (np.sum(XV ** q, axis=1) / ax1) ** (1 / q)
    return mq_array


def findF(X, lX, maxF, minF):
    range_x = np.where(np.logical_and(X >= minF, X <= maxF))[0]  # 7.3
    # range_x = np.where((X >= minF) & (X <= maxF))[0] #7.5
    return len(range_x) / lX


def findFdelta(X, lX, maxF, minF):
    delta = 0
    for x in X:
        if not (minF <= x <= maxF):
            piX1 = abs(x - minF)
            piX2 = abs(x - maxF)
            minPix = min(piX1, piX2)
            maxPix = max(piX1, piX2)
            delta += (minPix / maxPix)
    return delta / lX




def length_2point(XX, X, V, F, delta=False):
    XV = np.zeros((len(X), len(V)))
    lengthAx = len(XX)
    for f in range(len(F)):
        XXf = XX[:, f]
        for iX, x in enumerate(X[:, f]):
            if delta:
                xi_array = [1 - findFdelta(XXf, lengthAx, max(x, v), min(x, v)) for v in V[:, f]]
            else:
                xi_array = [findF(XXf, lengthAx, max(x, v), min(x, v)) for v in V[:, f]]
            XV[iX] += xi_array
    return XV

def barLength_2point(XX, X, V, F, delta=False):
    XV = np.zeros((len(X), len(V)))
    lengthAx = len(XX)
    for feat_group in F:
        for iX, x in enumerate(X):
            if delta:
                xi_array = [1 - (findFbarDelta(XX, x, v, feat_group) / lengthAx) for v in V]
            else:
                xi_array = [findFbar(XX, x, v, feat_group) / lengthAx for v in V]
            XV[iX] += xi_array
    return XV / len(F)

def findFbar(X, x, v, feat):
    Fx = 0
    for f in feat:
        maxF = max(x[f], v[f])
        minF = min(x[f], v[f])
        range_x = np.where(np.logical_and(X[:, f] >= minF, X[:, f] <= maxF))[0]
        Fx += len(range_x)
    return Fx / len(feat)


def findFbarDelta(X, x, v, feat):
    Fx = 0
    for xx in X:
        for f in range(feat):
            maxF = max(x[f], v[f])
            minF = min(x[f], v[f])
            if not (minF <= xx[f] <= maxF):
                piX1 = abs(xx[f] - minF)
                piX2 = abs(xx[f] - maxF)
                minPix = min(piX1, piX2)
                maxPix = max(piX1, piX2)
                Fx += (minPix / maxPix)
    return Fx / len(feat)


class Core:
    def __init__(self, X, Y, V, param, feats):
        self.X = X
        self.Y = Y
        self.V = V
        self.feats = feats
        self.param = param

        self.VV = self.learning()
        self.XV = self.calc_XV()

        self.idxB = self.alpha_parser(self.XV)

    def learning(self):
        if self.param.bar is True:
            learnV = barLength_2point(self.Y, self.V, self.V, self.feats, self.param.delta)
        else:
            learnV = length_2point(self.Y, self.V, self.V, self.feats, self.param.delta)
        return learnV

    def calc_XV(self):
        if self.param.bar is True:
            XV = barLength_2point(self.Y, self.X, self.V, self.feats, self.param.delta)
        else:
            XV = length_2point(self.Y, self.X, self.V, self.feats, self.param.delta)
        if len(XV[0]) > 1:
            XV = mq_axis1(XV, self.param.q)
        return XV

    def alpha_parser(self, XV):
        if self.param.alphaMax:
            '''alpha по границе V(V)'''
            Mq_learnV = mq_axis1(self.VV, self.param.q)
            alpha = max(Mq_learnV)
            idxB = np.where(self.XV <= alpha)[0]
            return idxB
        elif type(self.param.pers) is int:
            '''процент от XV'''
            X = np.ravel(XV)
            idxB = persRunner(X, self.param.pers, revers=False)
            return idxB
        elif self.param.kmeans is not False:
            '''kmeans кластер с наименьшим центроидом'''
            idxB = km(XV, self.param.kmeans, randCZ=False)[0]
            return np.array(idxB).astype(int)
        else:
            '''разделение по s степенному среднему'''
            MqXV = np.ravel(XV)
            s = self.param.s
            if s is None:
                print('Error\nFalse alpha param  s is None')
                alpha = None
            else:
                alpha = (np.sum(MqXV ** s) / len(MqXV)) ** (1 / s)
            idxB = np.where(XV <= alpha)[0]
            return idxB
