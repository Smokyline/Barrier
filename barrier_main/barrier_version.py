import multiprocessing as mp

from barrier_main.core import Core
from barrier_modules.tools import *


class BarrierMod:
    def __init__(self, imp, gp):
        self.imp = imp
        self.X = imp.data_full
        self.Y = imp.data_field
        self.V = imp.data_sample

        self.param = gp
        self.feats = gp.global_feats()


    def feats_run(self, que, v):
        idxXvF = np.array([]).astype(int)
        alpha_const_vF = []  # константы, разделяющие множество расстояний v малого f малого
        roXvF = []  # расстояния X до v малого F большое
        for fi, feat in enumerate(self.feats):
            XF = self.X[:, feat]
            YF = self.Y[:, feat]
            # VF = np.array([v])[:, feat]
            VF = v[feat]

            res = Core(XF, YF, VF, self.param, feat)
            idxXvF = np.append(idxXvF, res.idxB)
            roXvF.append(res.XV)
            alpha_const_vF.append(res.alpha_const)

        '''вычисление кол-ва попаданий Х в вс класс по константе alpha(v, F)'''
        countX = calc_count(idxXvF, roXvF, len(self.X), self.param.nchCount, alpha_const_vF)

        '''разделение кол-ва попаданий X по F большому в вс класс по border'''
        idxB, countX_const = count_border_blade(self.param.border, countX, border_const=None)
        que.put([countX, countX_const, idxB])


    def oneVoneP(self):
        V_queue = []
        for vi, v in enumerate(self.V):

            v_que = mp.Queue()
            p = mp.Process(target=self.feats_run, args=(v_que, v))
            V_queue.append(v_que)
            p.start()

        V_queue = np.array([que.get() for que in V_queue])
        countX_VF = np.ravel(V_queue[:, 0])
        countX_const_arr = np.ravel(V_queue[:, 1])
        full_idxB = np.array([]).astype(int)  # высокосейсмичные индексы
        for idxxxx in V_queue[:, 2]:
            full_idxB = np.append(full_idxB, idxxxx)

        final_idxB = np.unique(full_idxB).astype(int)
        return Result(final_idxB, countX_VF, countX_const_arr, self.param, self.imp, 'oneVoneP')

    def oneVoneP_Y(self):
        alpha_const_Y_arr = []  # константы, разделяющие множество расстояний B(Y, v, F)
        countY_const_arr = []  # константы, разделяющие множество попаданий Y в высокосейсмичный класс

        for vi, v in enumerate(self.V):
            idxYVF = np.array([]).astype(int)  # индексы Y попавших в вс класс по v малому F большому
            alpha_const_vF = []  # константы alpha, разделяющие расстояний от Y до v малому по f малому
            roYvF = []  # расстояния от Y до v малому F большому
            for fi, feat in enumerate(self.feats):
                YF = self.Y[:, feat]
                VF = v[feat]

                res = Core(YF, YF, VF, self.param, feat)
                idxYVF = np.append(idxYVF, res.idxB)
                roYvF.append(res.XV)
                alpha_const_vF.append(res.alpha_const)

            alpha_const_Y_arr.append(alpha_const_vF)
            '''вычисление кол-ва Y, попавших в вс класс по v малому при alpha константе'''
            countY = calc_count(idxYVF, roYvF, len(self.Y), self.param.nchCount, alpha_const_vF)
            '''вычисление константы, разделяющей кол-во попаданий Y по F большому, на вс и нс класс'''
            _, border_const_Y = count_border_blade(self.param.border, countY)
            countY_const_arr.append(border_const_Y)

        full_idxB = np.array([]).astype(int)
        countX_VF = []  # кол-во попаданий X в вс класс при v малом по признакам
        for vi, v in enumerate(self.V):
            idxXvF = np.array([]).astype(int)
            roXvF = []
            for fi, feat in enumerate(self.feats):
                XF = self.X[:, feat]
                YF = self.Y[:, feat]
                VF = v[feat]
                res = Core(XF, YF, VF, self.param, feat, alpha=alpha_const_Y_arr[vi][fi])

                idxXvF = np.append(idxXvF, res.idxB)
                roXvF.append(res.XV)

            '''вычисление кол-ва Y, попавших в вс класс по v малому при alpha(Y, v) константе'''
            countX = calc_count(idxXvF, roXvF, len(self.X), self.param.nchCount, alpha_const_Y_arr[vi])
            countX_VF.append(countX)

            '''выделение Y, попавший в вс класс по кол-во попаданий по признакам при v малом'''
            idxXv = count_border_blade(self.param.border, countX, border_const=countY_const_arr[vi])
            full_idxB = np.append(full_idxB, idxXv)

        final_idxB = np.unique(full_idxB).astype(int)
        return Result(final_idxB, countX_VF, countY_const_arr, self.param, self.imp, 'oneVoneP_Y')

    def union_nch_count_res(self, RE_count, RE_mean_const, gp, imp, theta):
        shape = RE_count.shape  # (len res, len v , len X)
        full_idx = np.array([]).astype(int)
        for vi in range(shape[1]):
            count_vi_R = RE_count[:, vi, :]
            const_mean_vi_R = RE_mean_const[:, vi]

            count_nch_vi = np.zeros((1, shape[2]))
            for r in range(shape[0]):
                const = const_mean_vi_R[r]
                nch = np.array([min(cnt, const) / const for cnt in np.ravel(count_vi_R[r])])
                count_nch_vi += nch
            idx, _ = count_border_blade(theta, count_nch_vi[0])
            full_idx = np.append(full_idx, idx)
        final_idx = np.unique(full_idx).astype(int)
        gp.border = theta
        gp.s = 'best'
        return Result(final_idx, None, None, gp, imp, 'nch_sum')
