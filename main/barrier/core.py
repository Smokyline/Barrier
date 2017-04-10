from main.barrier.test_foo import *
from main.alghTools.tools import pers_separator, found_nch_param_border
from main.alghTools.kmeans import km
import time

def mq_axis1(XV, q):
    """степенное среднее каждой строки """
    if q is None:
        print('Error\nq is None or v!=1')
        mq_array = None
    else:
        mq_array = np.array([])
        for xv in XV:
            xv = xv[np.where(xv != 0)]
            mq_xv = np.mean(xv ** q) ** (1 / q)
            mq_array = np.append(mq_array, [mq_xv])
    return mq_array


def simple_range(Y, X, V):
    """Вычисление расстояний между X и V
    признак f - вещественное число """

    def calc_range(yF, maxF, minF):
        """расстояние между x и v основано на колличестве элементов из Y
            с таким же параметром
            B(x, v, f) > 0 """
        #range_x = len(np.where(np.logical_and(yF >= minF, yF <= maxF))[0]) + 1 # 190ms
        range_x = np.count_nonzero((yF > minF) & (yF < maxF)) + 1 #90ms

        return range_x / lengthY

    lengthY = len(Y)
    lengthX = len(X)

    Xv_min = X.copy()
    Xv_max = X.copy()
    Xv_min[np.where(X > V)] = V
    Xv_max[np.where(X < V)] = V
    XV = np.array([calc_range(Y, Xv_max[xi], Xv_min[xi]) for xi in range(lengthX)])

    return XV


def vector_range(Y, X, V, F, delta=False):
    """Вычисление расстояний между X и V
    признак f - вектор """
    length_Y = len(Y)
    length_F = len(F)


    def calc_range(yF, xF, vF, length_feat):
        """расстояние между x и v основано на колличестве элементов из Y
                с таким же параметром
                B(x, v, F) > 0 """
        Fx = 0
        for f in range(length_feat):

            maxF = max(xF[f], vF[f])
            minF = min(xF[f], vF[f])

            #range_x = len(np.where(np.logical_and(yF[:, f] >= minF, yF[:, f] <= maxF))[0]) + 1 #720
            range_x = np.count_nonzero((minF < yF[:, f]) & (yF[:, f] < maxF)) + 1  # 340ms

            Fx += range_x / length_Y
        return Fx / length_feat

    XV = np.array([calc_range(Y, x, V, length_F) for x in X])
    return XV


class Core:
    def __init__(self, X, Y, V, param, feats, alpha=None):
        self.X = X  # объекты распознавания
        self.Y = Y  # объекты, по которым считается расстояние между X и V
        self.V = V  # объекты обучения
        self.feats = feats  # признаки
        self.param = param  # класс с константами

        '''вычисление расстояний между X и V на основе Y'''
        self.XV = self.calc_XV()

        '''Разделение множества X'''
        self.alpha_const = None
        self.idxB = self.alpha_parser(self.XV, alpha)



    def calc_VV(self):
        """Вычисление расстояний между V и V для нахождения минимального alpha порога
            если параметр alphaMax=True """
        if self.param.vector is True:
            learnV = vector_range(self.Y, self.V, self.V, self.feats, self.param.delta)
        elif self.param.metrix is True:
            learnV = metrix_length_2point(self.Y, self.V, self.V)
        else:
            learnV = simple_range(self.Y, self.V, self.V)

        if len(learnV[0]) > 1:
            learnV = mq_axis1(learnV, self.param.q)
        return learnV

    def calc_XV(self):
        """Вычисление расстояний между X и V """
        if self.param.vector is True:
            XV = vector_range(self.Y, self.X, self.V, self.feats, self.param.delta)
        elif self.param.metrix is True:
            XV = metrix_length_2point(self.Y, self.X, self.V)
        else:
            XV = simple_range(self.Y, self.X, self.V)
        return XV

    def alpha_parser(self, XV, alpha):

        if alpha is not None:  # если alpha уже вычисленно
            idxB = np.where(XV <= alpha)[0]
            return idxB

        elif self.param.alphaMax:
            '''alpha по границе V(V)'''
            Mq_learnV = mq_axis1(self.calc_VV(), self.param.q)
            self.alpha_const = max(Mq_learnV)
            idxB = np.where(self.XV <= self.alpha_const)[0]
            return idxB

        elif type(self.param.pers) is not False:
            '''процент от XV'''
            X = np.ravel(XV)
            idxB, self.alpha_const = pers_separator(X, self.param.pers, lower=True)
            return idxB

        elif self.param.kmeans is not False:
            '''kmeans кластер с наименьшим центроидом'''
            idxB, parsed_XV, self.alpha_const = km(XV, self.param.kmeans, randCZ=False)[0]
            return np.array(idxB).astype(int)
        elif self.param.beta or self.param.mcos is not False:
            MqXV = np.ravel(XV)
            idxB, self.alpha_const = found_nch_param_border(np.abs(MqXV-1), self.param.beta, self.param.mcos)
            return idxB
        else:
            '''разделение по s степенному среднему'''
            MqXV = np.ravel(XV)
            s = self.param.s
            if s is None:
                print('Error\nFalse alpha param  s is None')
            else:

                self.alpha_const = np.mean(MqXV ** s) ** (1 / s)

            idxB = np.where(XV <= self.alpha_const)[0]
            return idxB
