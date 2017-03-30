from main.barrier.core import Core
from main.supportAlg.drawMap import check_pix_pers
from main.alghTools.tools import *


class Result:
    def __init__(self, idxB, countX, countX_const_arr, param, imp, alg_name):
        self.result = idxB  # индексы высокосейсмичных узлов
        self.lenB = len(idxB)
        self.lenf = len(param.global_feats())

        self.countX = countX  # кол-во попаданий по признакам
        self.psi_mean_const = countX_const_arr

        self.pers = check_pix_pers(imp.data_coord[self.result], grid=imp.gridVers)  # процент занимаемой прощади
        self.acc = acc_check(imp.data_coord[self.result], imp.eq_all, grid=imp.gridVers)  # точность

        self.alg_name = alg_name
        self.param_title = set_title_param(vars(param))
        self.title = '%s B=%s(%s%s) acc=%s f=%s %s' % (self.alg_name, self.lenB, self.pers, '%', self.acc,
                                                       self.lenf, self.param_title)

class BarrierMod:
    def __init__(self, imp, g_param):
        self.imp = imp
        self.X = imp.data_full
        self.Y = imp.data_field
        self.V = imp.data_sample

        self.param = g_param
        self.feats = g_param.global_feats()

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

    def oneVoneP(self):
        full_idxB = np.array([]).astype(int)  # высокосейсмичные индексы
        countX_VF = []  # кол-во попаданий X в вс класс при v малом по признакам
        countX_const_arr = []  # константы границ, разделяющие кол-во попаданий по признакам
        for vi, v in enumerate(self.V):
            idxXvF = np.array([]).astype(int)
            alpha_const_vF = []  # константы, разделяющие множество расстояний v малого f малого
            roXvF = []  # расстояния X до v малого F большое
            for f in self.feats:
                feat = [f, ]
                XF = self.X[:, feat]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]
                res = Core(XF, YF, VF, self.param, feat)
                idxXvF = np.append(idxXvF, res.idxB)
                roXvF.append(res.XV)
                alpha_const_vF.append(res.alpha_const)

            # save_xv_to_csv(roXVF, vi)
            '''вычисление кол-ва попаданий Х в вс класс по константе alpha(v, F)'''
            countX = calc_count(idxXvF, roXvF, len(self.X), self.param.nchCount, alpha_const_vF)
            countX_VF.append(countX)

            '''разделение кол-ва попаданий X по F большому в вс класс по border'''
            idxB, countX_const = count_border_blade(self.param.border, countX, border_const=None)
            countX_const_arr.append(countX_const)

            full_idxB = np.append(full_idxB, idxB)

        final_idxB = np.unique(full_idxB).astype(int)
        return Result(final_idxB, countX_VF, countX_const_arr, self.param, self.imp, 'oneVoneP')

    def oneVoneP_Y(self):
        alpha_const_Y_arr = []  # константы, разделяющие множество расстояний B(Y, v, F)
        countY_const_arr = []  # константы, разделяющие множество попаданий Y в высокосейсмичный класс

        for vi, v in enumerate(self.V):
            idxYVF = np.array([]).astype(int)  # индексы Y попавших в вс класс по v малому F большому
            alpha_const_vF = []  # константы alpha, разделяющие расстояний от Y до v малому по f малому
            roYvF = []  # расстояния от Y до v малому F большому
            for fi, f in enumerate(self.feats):
                feat = [f, ]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]
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
            for fi, f in enumerate(self.feats):
                feat = [f, ]
                XF = self.X[:, feat]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]

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
