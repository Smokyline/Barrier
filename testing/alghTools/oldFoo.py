
def length_2point(XX, X, V, F, p, delta=False, sumP=True):
    XV = np.empty((len(F), len(X), len(V)))
    lengthAx = len(XX)
    for f in range(len(F)):
        XXf = XX[:, f]
        for iX, x in enumerate(X[:, f]):
            for iV, v in enumerate(V[:, f]):
                minF = min(x, v)
                maxF = max(x, v)
                if delta:
                    Fx = 1 - (findFdelta(XX[:, f], maxF, minF) / lengthAx)
                else:
                    Fx = findF(XXf, maxF, minF) / lengthAx
                XV[f, iX, iV] = Fx
    if sumP:
        pFEAT = (np.sum(XV ** p, axis=0) / len(F)) ** (1 / p)
    else:
        pFEAT = XV
    return pFEAT

def length_2point(XX, X, V, F, p, delta=False, sumP=True):
    XV = np.zeros((len(X), len(V)))
    lengthAx = len(XX)
    for f in range(len(F)):
        XXf = XX[:, f]
        for iX, x in enumerate(X[:, f]):
            xi_array = []
            for iV, v in enumerate(V[:, f]):
                minF = min(x, v)
                maxF = max(x, v)
                if delta:
                    Fx = 1 - (findFdelta(XX[:, f], maxF, minF) / lengthAx)
                else:
                    Fx = findF(XXf, maxF, minF) / lengthAx
                xi_array.append(Fx)
            XV[iX] += np.array(xi_array)

    return XV