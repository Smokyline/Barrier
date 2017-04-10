
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


def found_nch_param_border(X, beta, count=False):
    def mq(X, s):
        mq_mean = np.mean(X ** s) ** (1 / s)
        return mq_mean

    if count:
        mq_power = np.arange(1, 51, 1)
    else:
        mq_power = np.arange(-50, 0, 0.5)

    X = X[np.where(X != 0)]
    mq_value = [mq(X, s) for s in mq_power]
    deriv = np.array([mq_value[i + 1] - mq_value[i] for i in range(len(mq_value) - 1)])
    alpha = calc_nch_alpha(deriv, beta=beta)
    near_alpha_drv = np.argmin(np.abs(deriv - alpha))
    border = mq_value[near_alpha_drv]

    if count:
        idx = np.where(X >= border)[0]
    else:
        idx = np.where(X <= border)[0]
    return idx, border