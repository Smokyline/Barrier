import numpy as np
import multiprocessing as mp
from module import tools


class Barrier:
    def __init__(self, X, Y, param):
        self.X = X
        self.Y = Y
        self.param = param

        self.lenX, self.lenf = np.shape(X)

        self.MY = self.matrix_XY()
        self.hs_indexes_unionV, self.hs_indexes_v = self.recog_hs_idx(self.MY)
        self.hs_F_top, self.hs_F_count = self.recog_hs_feats(self.MY)

    def recog_hs_feats(self, MY):

        def sigma(Y, lenf):
            S = np.zeros((1, lenf))[0]
            for y in Y:
                for idx in y:
                    S[idx] += 1
            return S.astype(int)

        C = np.zeros((1, self.lenf))[0]  # кол-во высокосейсмичных узлов
        Z = []  # кол-во попаданий в топ

        for idx_v, My in enumerate(MY):
            ro_My = np.sum(My[self.hs_indexes_v[idx_v], :], axis=0)  # кол-во попаданий признака в В класс
            sort_idx = np.argsort(ro_My)  # from lowest to highest

            """нахождение высокосейсмичных признаков по кол-ву объектов в В классе"""
            for p in sort_idx[-self.param.omega:]:
                C[p] += ro_My[p]
            #C += ro_My


            """нахождение высокосейсмичных признаков по кол-ву попаданий в топ"""
            Z.append(sort_idx[-self.param.omega:])

        Z = sigma(Z, self.lenf)

        return np.array(Z), C/len(self.Y)

    def recog_hs_idx(self, MY):
        """нахождение высокосейсмичных объектов во множестве Х"""

        hs_idx_v = []
        hs_idx_unionV = []
        for My in MY:
            ro_My = np.sum(My, axis=1)  # кол-во попаданий х в В класс по y
            idx_hs, _ = tools.count_border_blade(self.param.border, ro_My)
            hs_idx_v.append(idx_hs)
            hs_idx_unionV.extend(idx_hs)

        return np.unique(hs_idx_unionV), hs_idx_v

    def matrix_Xy(self, que, y):
        """матрица сейсмичности Х к объекту обучения по всем признакам """
        M = np.empty((self.lenf, self.lenX))  # lenF; lenX
        for f in range(self.lenf):
            measure_Xyf = self.measure(self.X[:, f], y[f])
            bool_hs_array = self.alpha_parse_measure(measure_Xyf)
            M[f] = bool_hs_array
        que.put(M.transpose())

    def matrix_XY(self):
        """матрицы сейсмичности Х ко всем объектам обучения по все признакам"""

        Y_matrix = []
        for y_start in range(0, len(self.Y), 8):
            core_que = []
            for yi in np.arange(len(self.Y))[y_start:y_start + 8]:
                "близость объектов к объекту обучения по всем признакам"
                que = mp.Queue()
                p = mp.Process(target=self.matrix_Xy, args=(que, self.Y[yi]))
                core_que.append(que)
                p.start()
            core_que = [qe.get() for qe in core_que]
            Y_matrix.extend(core_que)

        return np.array(Y_matrix)


    def measure(self, X, y):
        """Вычисление расстояний между X и V
        признак f - вещественное число """

        def calc_range(Xf, maxF, minF):
            """расстояние между x и v основано на колличестве элементов из Y
                с таким же параметром
                B(x, v, f) > 0 """
            range_x = np.count_nonzero((Xf >= minF) & (Xf <= maxF))
            return range_x / len(Xf)

        Xy_max = np.maximum(X, y)
        Xy_min = np.minimum(X, y)
        measure_Xy = np.array([calc_range(X, Xy_max[xi], Xy_min[xi]) for xi in range(self.lenX)])
        return measure_Xy

    def alpha_parse_measure(self, measure_Xy):
        """разделение по s степенному среднему"""
        MqXy = np.ravel(measure_Xy)
        s = self.param.s
        if s is None:
            alpha_const = None
            print('Error\nFalse alpha param  s is None')
        else:
            alpha_const = np.mean(MqXy ** s) ** (1 / s)
        idxB_Xfy = np.where(MqXy <= alpha_const)[0]

        bool_B_Xy = np.zeros((1, self.lenX))[0]
        for i in range(self.lenX):
            if i in idxB_Xfy:
                bool_B_Xy[i] = 1
        return bool_B_Xy
