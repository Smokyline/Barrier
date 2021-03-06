from module import tools
import numpy as np
import multiprocessing as mp

class Core:
    def __init__(self, X, Y, V, param, feats):
        self.X = X  # объекты распознавания
        self.Y = Y  # объекты, по которым считается расстояние между X и V
        self.V = V  # объекты обучения
        self.feats = feats  # признаки
        self.param = param  # класс с константами

        '''вычисление расстояний между X и V на основе Y'''
        self.XV = self.calc_XV()

        '''Разделение множества X'''
        self.alpha_const = None
        self.idxB = self.alpha_parser(self.XV)

    def calc_XV(self):
        """Вычисление расстояний между X и V """
        XV = self.count_range(self.Y, self.X, self.V)
        return XV

    def alpha_parser(self, XV):
        """разделение по s степенному среднему"""
        MqXV = np.ravel(XV)
        s = self.param.s
        if s is None:
            print('Error\nFalse alpha param  s is None')
        else:
            self.alpha_const = np.mean(MqXV ** s) ** (1 / s)

        idxXvf = np.where(XV <= self.alpha_const)[0]
        return idxXvf

    def count_range(self, Y, X, V):
        """Вычисление расстояний между X и V
        признак f - вещественное число """

        def calc_range(YF, maxF, minF):
            """расстояние между x и v основано на колличестве элементов из Y
                с таким же параметром
                B(x, v, f) > 0 """
            range_x = np.count_nonzero((YF >= minF) & (YF <= maxF))

            return range_x / lengthY

        lengthY = len(Y)
        lengthX = len(X)

        Xv_max = np.maximum(X, V)
        Xv_min = np.minimum(X, V)

        XV = np.array([calc_range(Y, Xv_max[xi], Xv_min[xi]) for xi in range(lengthX)])
        return XV



class Barrier:
    def __init__(self, imp, gp):
        self.alg_name = 'Barrier'

        self.imp = imp
        self.X = imp.data_full
        self.Y = imp.data_field
        self.V = imp.data_sample

        self.param = gp
        self.feats = gp.global_feats()


    def feats_and_v(self, que, v):
        idxXvF = np.array([]).astype(int)
        roXvF = []  # расстояния X до v малого F большое
        countBinF = np.zeros((1, len(self.feats)))[0]  # кол-во B в признаке
        for fi, feat in enumerate(self.feats):
            XF = self.X[:, feat]
            YF = self.Y[:, feat]
            VF = v[feat]

            core = Core(XF, YF, VF, self.param, feat)
            idxXvF = np.append(idxXvF, core.idxB)
            roXvF.append(core.XV)



        def calc_count(full_XvF_count, lX):
            """вычисление кол-ва попаданий X в вс класс по v малому f малому"""
            full_XvF_count = np.ravel(full_XvF_count)
            countX = np.array([len(np.where(full_XvF_count == i)[0]) for i in range(lX)]).astype(int)
            return countX

        '''вычисление кол-ва попаданий Х в вс класс по v малому'''
        countX = calc_count(idxXvF, len(self.X))
        print('-------------')

        ######################################
        '''разделение кол-ва попаданий X по F большому в вс класс по border'''
        idxB, countX_const = tools.count_border_blade(self.param.border, countX)
        que.put([countX, countX_const, idxB])


    def sample_union(self):
        V_measure = []
        for vi, v in enumerate(self.V):
            "близость объектов к объекту обучения по всем признакам"
            v_que = mp.Queue()
            p = mp.Process(target=self.feats_and_v, args=(v_que, v))
            V_measure.append(v_que)
            p.start()

        # кол-во попаданий X в hs,
        V_measure = np.array([que.get() for que in V_measure])

        array_of_hs_idx_v = np.array([]).astype(int)  # высокосейсмичные индексы
        for hs_idx_for_v in V_measure[:, 2]:
            array_of_hs_idx_v = np.append(array_of_hs_idx_v, hs_idx_for_v)

        self.count_of_hs_obj = np.ravel(V_measure[:, 0])

        self.hs_indexes = np.unique(array_of_hs_idx_v).astype(int)



