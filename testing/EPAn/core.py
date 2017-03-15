import numpy as np
from testing.alghTools.tools import *
import time
import math



def mq_axis1(XV, q):
    """степенное среднее каждой строки"""
    if q is None:
        print('Error\nq is None or v!=1')
        mq_array = None
    else:
        mq_array = np.array([])
        for xv in XV:
            idx_wh = np.where(xv != 0)[0]
            xv = xv[idx_wh]
            mq_xv = np.mean(xv ** q) ** (1 / q)
            mq_array = np.append(mq_array, [mq_xv])
    return mq_array

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

def barLength_2point(XX, X, V, F, delta=False):
    XV = np.zeros((len(X), len(V)))
    lengthAx = len(XX)
    for jf, feat_group in enumerate(F):
        for iX, x in enumerate(X):
            if delta:
                xi_array = [1 - (findFbarDelta(XX[:, jf], x[jf], v[jf], feat_group) / lengthAx) for v in V]
            else:
                xi_array = [findFbar(XX[:, jf], x[jf], v[jf], feat_group) / lengthAx for v in V]
            XV[iX] += xi_array
    return XV

def findFbar(X, x, v, feat):
    Fx = 0
    for f in range(len(feat)):
        maxF = max(x[f], v[f])
        minF = min(x[f], v[f])
        range_x = len(np.where(np.logical_and(X[:, f] >= minF, X[:, f] <= maxF))[0])
        Fx += (range_x / len(feat))
    return Fx



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

def metrix_length_2point(XX, X, V):

    XV = np.zeros((len(X), len(V)))
    len_x = XX.shape[0]
    for iX, x in enumerate(X):
        xi_array = [findF_metrix(XX, len_x, x[0], v[0]) for v in V]
        XV[iX] += xi_array
    return XV

def findF_metrix(Y, len_y, x, v):

    dxv = np.sum(np.abs(x - v))
    dxy = np.sum(np.abs(x - Y), axis=2)
    dvy = np.sum(np.abs(v - Y), axis=2)
    dy = np.amax(np.append(dvy, dxy, axis=1), axis=1)
    ro_xv = len(np.where(dy <= dxv)[0])

    return ro_xv/len_y



class Core:
    def __init__(self, X, Y, V, param, feats, alpha=None):
        self.X = X
        self.Y = Y
        self.V = V
        self.feats = feats
        self.param = param

        self.XV = self.calc_XV()

        self.alpha_const = None
        self.idxB = self.alpha_parser(self.XV, alpha)

    def calc_VV(self):
        if self.param.bar is True:
            learnV = barLength_2point(self.Y, self.V, self.V, self.feats, self.param.delta)
        elif self.param.metrix is True:
            learnV = metrix_length_2point(self.Y, self.V, self.V)

        else:
            learnV = length_2point(self.Y, self.V, self.V, self.feats, self.param.delta)
        if len(learnV[0]) > 1:
            learnV = mq_axis1(learnV, self.param.q)
        return learnV

    def calc_XV(self):
        if self.param.bar is True:
            XV = barLength_2point(self.Y, self.X, self.V, self.feats, self.param.delta)
        elif self.param.metrix is True:
            XV = metrix_length_2point(self.Y, self.X, self.V)
        else:
            XV = length_2point(self.Y, self.X, self.V, self.feats, self.param.delta)
        if len(XV[0]) > 1:
            XV = mq_axis1(XV, self.param.q)
        return XV

    def alpha_parser(self, XV, alpha):
        if alpha is not None:
            idxB = np.where(XV <= alpha)[0]
            return idxB
        elif self.param.alphaMax:
            '''alpha по границе V(V)'''
            Mq_learnV = mq_axis1(self.calc_VV(), self.param.q)
            self.alpha_const = max(Mq_learnV)
            idxB = np.where(self.XV <= self.alpha_const)[0]
            return idxB
        elif type(self.param.pers) is int:
            '''процент от XV'''
            X = np.ravel(XV)
            idxB, self.alpha_const = persRunner(X, self.param.pers, revers=False)
            return idxB
        elif self.param.kmeans is not False:
            '''kmeans кластер с наименьшим центроидом'''
            idxB, self.alpha_const = km(XV, self.param.kmeans, randCZ=False)[0]
            return np.array(idxB).astype(int)
        else:
            '''разделение по s степенному среднему'''
            MqXV = np.ravel(XV)
            s = self.param.s
            if s is None:
                print('Error\nFalse alpha param  s is None')
            else:
                self.alpha_const = (np.sum(MqXV ** s) / len(MqXV)) ** (1 / s)
            idxB = np.where(XV <= self.alpha_const)[0]
            return idxB
