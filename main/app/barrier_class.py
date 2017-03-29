from main.barrier.core import Core
from main.supportAlg.drawMap import check_pix_pers
from main.alghTools.kmeans import km
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

    def count_border_blade(self, border, countX, border_const=None):
        """h(V); h(X); pers; kmeans"""

        if border_const is not None:
            idxXV = np.array(np.where(countX >= border_const)[0]).astype(int)
            return idxXV
        else:
            if border[0] == 'h(X)':
                idxXV, const = h_separator(len(countX), countX, h=border[1])
            elif border[0] == 'ro':
                idxXV, const = ro_separator(len(countX), countX, r=border[1])
            elif border[0] == 'pers':
                idxXV, const = pers_separator(countX, pers=border[1], revers=True)
            elif border[0] == 'kmeans':
                _, idxXV, const = km(countX, border[1], randCZ=False)


            else:
                print('wrong border')
                idxXV, const = None, None
            return np.array(idxXV).astype(int), const

    def calc_count(self, idxs, XVF, lX, nch=False, alpha_array=None):
        if not nch:
            idxs = np.ravel(idxs)
            return np.array([len(np.where(idxs == i)[0]) for i in range(lX)]).astype(int)
        else:
            count_Xi = np.zeros((1, lX))
            for i, XV in enumerate(XVF):
                alpha_i = alpha_array[i]
                nch_XV = np.array([alpha_i / (max(alpha_i, xv)) for xv in np.ravel(XV)])
                count_Xi += nch_XV
            return count_Xi[0]

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
            idx, _ = self.count_border_blade(theta, count_nch_vi[0])
            full_idx = np.append(full_idx, idx)
        final_idx = np.unique(full_idx).astype(int)
        gp.border = theta
        gp.s = 'best'
        return Result(final_idx, None, None, gp, imp, 'nch_sum')

    def oneVoneP(self):
        fullXV = np.array([]).astype(int)
        countXV = []
        countX_const_arr = []
        for vi, v in enumerate(self.V):
            idxXVF = np.array([]).astype(int)
            alpha_const_v = []
            roXVF = []
            for f in self.feats:
                feat = [f, ]
                XF = self.X[:, feat]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]
                res = Core(XF, YF, VF, self.param, feat)
                idxXVF = np.append(idxXVF, res.idxB)
                roXVF.append(res.XV)
                alpha_const_v.append(res.alpha_const)
            # save_xv_to_csv(roXVF, vi)
            countX = self.calc_count(idxXVF, roXVF, len(self.X), self.param.nchCount, alpha_const_v)
            countXV.append(countX)
            idxXv, countX_const = self.count_border_blade(self.param.border, countX, border_const=None)
            fullXV = np.append(fullXV, idxXv)
            countX_const_arr.append(countX_const)
        print(countX_const_arr)

        idxXV = np.unique(fullXV).astype(int)
        return Result(idxXV, countXV, countX_const_arr, self.param, self.imp, 'oneVoneP')

    def oneVoneP_Y(self):
        alpha_const_Y_array = []
        countY_const_array = []

        for vi, v in enumerate(self.V):
            idxYVF = np.array([]).astype(int)
            alpha_const_v = []
            roYVF = []
            for fi, f in enumerate(self.feats):
                feat = [f, ]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]
                res = Core(YF, YF, VF, self.param, feat)
                idxYVF = np.append(idxYVF, res.idxB)
                roYVF.append(res.XV)
                alpha_const_v.append(res.alpha_const)
            countY = self.calc_count(idxYVF, roYVF, len(self.Y), self.param.nchCount, alpha_const_v)
            idxYv, border_const_Y = self.count_border_blade(self.param.border, countY)
            alpha_const_Y_array.append(alpha_const_v)
            countY_const_array.append(border_const_Y)
        fullXV = np.array([]).astype(int)
        countXV = []
        for vi, v in enumerate(self.V):
            idxXVF = np.array([]).astype(int)
            roXVF = []
            for fi, f in enumerate(self.feats):
                feat = [f, ]
                XF = self.X[:, feat]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]
                res = Core(XF, YF, VF, self.param, feat, alpha=alpha_const_Y_array[vi][fi])
                idxXVF = np.append(idxXVF, res.idxB)
                roXVF.append(res.XV)

            countX = self.calc_count(idxXVF, roXVF, len(self.X), self.param.nchCount, alpha_const_Y_array[vi])
            countXV.append(countX)
            idxXv = self.count_border_blade(self.param.border, countX, border_const=countY_const_array[vi])
            fullXV = np.append(fullXV, idxXv)

        idxXV = np.unique(fullXV).astype(int)
        return Result(idxXV, countXV, countY_const_array, self.param, self.imp, 'oneVoneP_Y')
