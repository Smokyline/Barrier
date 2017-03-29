from main.barrier.test_foo import *
from main.alghTools.tools import pers_separator
from main.alghTools.kmeans import km


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


def simple_range(Y, X, V, F, delta=False):
    """Вычисление расстояний между X и V
    признак f - вещественное число """
    XV = np.zeros((len(X), len(V)))
    lengthY = len(Y)

    def calc_range(yF, maxF, minF):
        """расстояние между x и v основано на колличестве элементов из Y
            с таким же параметром
            B(x, v, f) > 0 """
        range_x = len(np.where(np.logical_and(yF >= minF, yF <= maxF))[0]) + 1
        return range_x / lengthY

    for f in range(len(F)):
        Yf = Y[:, f]
        for iX, x in enumerate(X[:, f]):
            if delta:
                xi_array = [1 - findFdelta(Yf, lengthY, max(x, v), min(x, v)) for v in V[:, f]]
            else:
                xi_array = [calc_range(Yf, max(x, v), min(x, v)) for v in V[:, f]]
            XV[iX] += xi_array
    return XV


def vector_range(Y, X, V, F, delta=False):
    """Вычисление расстояний между X и V
    признак f - вектор """
    XV = np.zeros((len(X), len(V)))
    length_Y = len(Y)

    def calc_range(yF, xF, vF, length_feat):
        """расстояние между x и v основано на колличестве элементов из Y
                с таким же параметром
                B(x, v, F) > 0 """
        Fx = 0
        for f in range(length_feat):
            maxF = max(xF[f], vF[f])
            minF = min(xF[f], vF[f])
            range_x = len(np.where(np.logical_and(yF[:, f] >= minF, yF[:, f] <= maxF))[0]) + 1
            Fx += range_x
        return Fx / length_feat

    for jf, feat_group in enumerate(F):
        for iX, x in enumerate(X):
            if delta:
                xi_array = [1 - (findFbarDelta(Y[:, jf], x[jf], v[jf], feat_group) / length_Y) for v in V]
            else:
                xi_array = [calc_range(Y[:, jf], x[jf], v[jf], len(feat_group)) for v in V]
            XV[iX] += xi_array

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
            learnV = simple_range(self.Y, self.V, self.V, self.feats, self.param.delta)

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
            XV = simple_range(self.Y, self.X, self.V, self.feats, self.param.delta)
        if len(XV[0]) > 1:
            XV = mq_axis1(XV, self.param.q)
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

        elif type(self.param.pers) is int:
            '''процент от XV'''
            X = np.ravel(XV)
            idxB, self.alpha_const = pers_separator(X, self.param.pers, revers=False)
            return idxB

        elif self.param.kmeans is not False:
            '''kmeans кластер с наименьшим центроидом'''
            idxB, parsed_XV, self.alpha_const = km(XV, self.param.kmeans, randCZ=False)[0]
            return np.array(idxB).astype(int)

        else:
            '''разделение по s степенному среднему'''
            MqXV = np.ravel(XV)
            s = self.param.s
            if s is None:
                print('Error\nFalse alpha param  s is None')
            else:
                self.alpha_const = mq_axis1(MqXV, s)
            idxB = np.where(XV <= self.alpha_const)[0]
            return idxB
