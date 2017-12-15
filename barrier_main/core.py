from barrier_modules.tools import *


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


def simple_range(Y, X, V, metrics=False):
    """Вычисление расстояний между X и V
    признак f - вещественное число """

    def calc_range(YF, maxF, minF):
        """расстояние между x и v основано на колличестве элементов из Y
            с таким же параметром
            B(x, v, f) > 0 """
        range_x = np.count_nonzero((YF >= minF) & (YF <= maxF))

        return range_x / lengthY


    def calc_metrics_range(Y, maxF, minF):
        delta = 0
        for y in Y:
            if not (minF <= y <= maxF):
                piX1 = abs(y - minF)
                piX2 = abs(y - maxF)
                minPix = min(piX1, piX2)
                maxPix = max(piX1, piX2)
                delta += (minPix / maxPix)
        return delta / lengthY

    lengthY = len(Y)
    lengthX = len(X)

    Xv_max = np.maximum(X, V)
    Xv_min = np.minimum(X, V)

    if metrics:
        XV = np.array([calc_metrics_range(Y, Xv_max[xi], Xv_min[xi]) for xi in range(lengthX)])
        XV = 1-XV
    else:
        XV = np.array([calc_range(Y, Xv_max[xi], Xv_min[xi]) for xi in range(lengthX)])
    return XV



def vector_range(Y, X, v, F, metrics=False):
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
            range_x = np.count_nonzero((minF < yF[:, f]) & (yF[:, f] < maxF)) + 1  # 340ms
            Fx += range_x / length_Y
        return Fx / length_feat

    def calc_metrics_range(YF, xF, vF, length_feat):
        Fx = 0
        for f in range(length_feat):
            maxF = max(xF[f], vF[f])
            minF = min(xF[f], vF[f])
            out_range_xv = np.where(np.logical_or((YF[:, f] < minF), (YF[:, f] > maxF)))[0]
            delta = 0

            if len(out_range_xv) != 0:
                out_Y = YF[out_range_xv, f]

                piX1 = np.abs(out_Y - minF)
                piX2 = np.abs(out_Y - maxF)

                xyv_max = np.maximum(piX1, piX2)
                xyv_min = np.minimum(piX1, piX2)
                delta = np.mean(xyv_min / xyv_max)

            Fx += delta
        return Fx / length_feat

    if metrics:
        XV = np.array([calc_metrics_range(Y, x, v, length_F) for x in X])
        XV = 1-XV
    else:
        XV = np.array([calc_range(Y, x, v, length_F) for x in X])
    return XV


def manhattan_range(X, v, F):

    def calc_range(xF, vF):
        Fx = np.mean(np.abs(xF-vF))
        return Fx

    XV = np.array([calc_range(x, v) for x in X])
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
            learnV = vector_range(self.Y, self.V, self.V, self.feats)
        else:
            learnV = simple_range(self.Y, self.V, self.V)

        if len(learnV[0]) > 1:
            learnV = mq_axis1(learnV, self.param.q)
        return learnV

    def calc_XV(self):
        """Вычисление расстояний между X и V """
        if self.param.vector is True:
            XV = vector_range(self.Y, self.X, self.V, self.feats, metrics=self.param.metrics)
        elif self.param.manhattan is True:
            XV = manhattan_range(self.X, self.V, self.feats)
        else:
            XV = simple_range(self.Y, self.X, self.V, metrics=self.param.metrics)
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

        elif self.param.pers is not False:
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
            idxB, self.alpha_const = found_nch_param_border(np.abs(1-MqXV), self.param.beta, self.param.mcos)
            return idxB
        elif self.param.range_const is not False:
            MqXV = np.ravel(XV)
            idxB = np.where(np.abs(1-MqXV) >= self.param.range_const)[0]
            self.alpha_const = self.param.range_const
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
