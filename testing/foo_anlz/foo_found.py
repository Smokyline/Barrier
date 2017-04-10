import numpy as np
import sys

def calc_a(X, beta):
    min_x = -0.9998
    max_x = 1.
    #epsl = 0.000000000006
    #epsl = 0.000006
    epsl = sys.float_info.epsilon
    X = X[np.where(X > 0)]

    def foo(B, a, beta):
        EPx = 0
        for b in B:
            EPx += (b - a) / max(a, b)
            #EPx += (a - b) / max(a, b)

        return (EPx / len(B)) - beta

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

        #if fA_min * fA_half < 0:
        #    max_x = half_x
        #else:
        #    min_x = half_x

        if fA_min * fA_half > 0:
            min_x = half_x
        else:
            max_x = half_x

        if max_x - min_x < epsl:
            alpha = half_x
            break

    return alpha
