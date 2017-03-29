import numpy as np

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