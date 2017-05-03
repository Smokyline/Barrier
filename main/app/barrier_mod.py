from main.barrier.core import Core
from main.supportAlg.drawMap import check_pix_pers, Visual
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
        save_res_idx_to_csv(imp.data_full, idxB, self.title)


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

        #vis = Visual(X=self.imp.data_coord, r=0.225, imp=self.imp, path=self.imp.save_path)
        for vi, v in enumerate(self.V):
            idxXvF = np.array([]).astype(int)
            alpha_const_vF = []  # константы, разделяющие множество расстояний v малого f малого
            roXvF = []  # расстояния X до v малого F большое
            for fi, feat in enumerate(self.feats):
                XF = self.X[:, feat]
                YF = self.Y[:, feat]
                #VF = np.array([v])[:, feat]
                VF = v[feat]

                res = Core(XF, YF, VF, self.param, feat)
                idxXvF = np.append(idxXvF, res.idxB)
                roXvF.append(res.XV)
                alpha_const_vF.append(res.alpha_const)

                gp_t = 'B3'
                #save_xv_to_csv(res.XV, [vi+1, fi+1], '%srange' % gp_t, 'XVrange')

                print('v%iп%i: %i' % (vi, fi, len(res.idxB)), end=' | ')

            '''вычисление кол-ва попаданий Х в вс класс по константе alpha(v, F)'''
            countX = calc_count(idxXvF, roXvF, len(self.X), self.param.nchCount, alpha_const_vF)
            countX_VF.append(countX)

            #save_xv_to_csv(np.mean(roXvF, axis=0), [vi, 0], '%srange' % gp_t, 'XVrange')
            #save_xv_to_csv(roXvF, [vi, 0], 'XVrange')

            #save_xv_to_csv(countX, [vi, 0], '%srange' % gp_t, 'count')

            '''разделение кол-ва попаданий X по F большому в вс класс по border'''
            idxB, countX_const = count_border_blade(self.param.border, countX, border_const=None)
            countX_const_arr.append(countX_const)
            #print(alpha_const_vF)
            #print(countX_const)
            print('\nlen v%iП: %i'%(vi, len(idxB)), '   alpha', alpha_const_vF, end='\n----------------------------\n')

            #vis.grid_res(self.imp.data_coord[idxB], title='v%i' % (vi+1), r=0.2)

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
