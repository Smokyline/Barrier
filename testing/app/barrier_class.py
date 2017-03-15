from testing.EPAn.core import *
from testing.alghTools.drawMap import *
import numpy as np



class Result:
    def __init__(self, idxB, param, imp, alg_name):
        self.result = idxB
        self.alg_name = alg_name
        #self.V = find_VinXV(imp.indexX, idxB, imp.indexV)
        self.V = np.array([[]])
        self.pers = check_pix_pers(imp.data_coord[self.result], grid=imp.gridVers)
        self.acc = acc_check(imp.data_coord[self.result], imp.eq_all, grid=imp.gridVers)
        self.param_title = set_title_param(vars(param))
        self.lenB = len(idxB)
        self.lenf = len(param.global_feats())
        self.title = '%s B=%s(%s%s) acc=%s f=%s %s' % (self.alg_name, self.lenB, self.pers, '%', self.acc,
                                                         self.lenf, self.param_title)


class CompareAlgh:
    def __init__(self, imp, vis, barrierX, coraX):
        self.algA = barrierX
        self.algB = coraX

        self.union = np.union1d(self.algA, self.algB)
        self.inters = list(set(self.algA) & set(self.algB))
        self.AwB = idx_diff_runnerAwB(self.algA, self.algB)
        self.BwA = idx_diff_runnerAwB(self.algB, self.algA)

        self.persUnion = check_pix_pers(imp.data_coord[self.union], grid=imp.gridVers)
        self.persA = check_pix_pers(imp.data_coord[self.algA], grid=imp.gridVers)
        self.persB = check_pix_pers(imp.data_coord[self.algB], grid=imp.gridVers)

        self.accA = acc_check(imp.data_coord[self.algA], imp.eq_all)
        self.accB = acc_check(imp.data_coord[self.algB], imp.eq_all)
        self.accUnion = acc_check(imp.data_coord[self.union], imp.eq_all)

        self.data_coord = imp.data_coord
        self.save_path = imp.save_path
        self.vis = vis

    def tanimoto(self):
        """мера Танимото (пересечение\объединение)"""
        return round(len(self.inters) / len(self.union), 2)

    def differ(self):
        """разность (объединение минус пересечение)"""
        return np.union1d(self.AwB, self.BwA)

    def visual_compare(self, result):
        compare_title = 'comp %s P=%s Bar(%s%s) vs Cora(%s%s) U=%s(%s%s) ' % (
            result.alg_name, result.lenf, self.persA, '%', self.persB, '%', len(self.union), self.persUnion, '%')
        compare_title2 = 'accBar=%s accCora=%s accU=%s tanim=%s BnC=%s B/C=%s C/B=%s' % (
            self.accA, self.accB, self.accUnion, self.tanimoto(), len(self.inters), len(self.AwB), len(self.BwA))

        self.vis.ln_diff_res(SETS=[self.data_coord[self.inters], self.data_coord[self.AwB], self.data_coord[self.BwA]], labels=['BnC', 'B/C', 'C/B'], title=compare_title, title2=compare_title2)
        self.vis.color_res(res=self.algB, title='Cora-3')

        self.vis.bw_stere_res(B=self.data_coord[self.algA], head_title='Барьер', circle_color='#aeaeae')
        self.vis.bw_stere_res(B=self.data_coord[self.algB], head_title='Кора-3', circle_color='none')
        self.vis.bw_stere_res(B=self.data_coord[self.union], head_title='Объединение', circle_color='#898989')

        visuaMSdiffPix_ras(self.data_coord[self.union], self.data_coord[self.inters], r=0.225, direc=self.save_path, title='Разность площадей')



class BarrierMod:
    def __init__(self, imp, g_param):
        self.imp = imp
        self.X = imp.data_full
        self.Y = imp.data_field
        self.V = imp.data_sample
        self.indexX = imp.indexX
        self.indexV = imp.indexV

        self.param = g_param
        self.feats = g_param.global_feats()

    def count_border_blade(self, border, countX, border_const=None):
        """h(V); h(X); pers; kmeans"""

        if border_const is not None:
            idxXV = np.array(np.where(countX >= border_const)[0]).astype(int)
            return idxXV
        else:
            if border[0] == 'h(X)':
                idxXV, const = parseIdxH(len(countX), countX, h=border[1])
            elif border[0] == 'ro':
                idxXV, const = parseIdx_ro(len(countX), countX, r=border[1])
            elif border[0] == 'pers':
                idxXV, const = persRunner(countX, pers=border[1], revers=True)
            elif border[0] == 'kmeans':
                idxXV, const = km(countX, border[1], randCZ=False)[-1]

            else:
                print('wrong border')
                idxXV, const = None, None
            return np.array(idxXV).astype(int), const

    def simple(self):
        X = self.X[:, self.feats]
        Y = self.Y[:, self.feats]
        V = self.V[:, self.feats]
        idxB = Core(X, Y, V, self.param, self.feats).idxB

        return Result(idxB, self.param, self.imp, 'simple')

    def iter_learning(self):
        X = self.X[:, self.feats]
        Y = self.Y[:, self.feats]
        V = self.V[:, self.feats]
        old_idx = None
        while True:
            idxB = Core(X, Y, V, self.param, self.feats).idxB
            V = X[idxB]
            if len(idxB) >= 130:
                break
            old_idx = idxB
        return Result(old_idx, self.param, self.imp, 'iterLearning')

    def adaXoneV(self):
        idxX = np.arange(len(self.X))
        idxV = find_VinXV(self.indexX, idxX, self.indexV)
        itr = 1
        while True:
            print('\niter', itr)
            print('|V|=%s |X|=%s' % (len(idxV), len(idxX)))
            idxNewX = np.array([])
            idxNewV = np.array([])
            idxXwV = np.array([int(i) for i, x in enumerate(idxX) if x not in idxV])

            for v in self.X[idxV]:
                fullXV = np.array([])
                for f in self.feats:
                    feat = [f, ]
                    XF = self.X[idxX][:, feat]
                    VF = np.array([v])[:, feat]
                    idxB = Core(XF, VF, self.param, feat).idxB
                    fullXV = np.append(fullXV, idxB)

                countX = np.array([len(np.where(fullXV == i)[0]) for i in range(len(idxX))])

                UPDcountX = countX[idxXwV]
                UPDidxX = idxX[idxXwV]
                UPDV = np.array([int(idx) for i, idx in enumerate(UPDidxX) if UPDcountX[i] >= max(UPDcountX)]).astype(int)

                idxNewX = np.append(idxNewX, UPDidxX).astype(int)
                idxNewV = np.append(idxNewV, UPDV)

            idxNewV = np.unique(idxNewV).astype(int)
            # idxX = idxXwV[[int(i) for i, x in enumerate(idxNewX) if x not in idxNewV]]
            idxVit = np.union1d(idxV, idxNewV)
            if len(idxVit) > 150:
                return Result(idxV, self.param, self.imp, 'adaXoneV')
            idxV = idxVit
            itr += 1

    def adaXallV(self):
        idxX = np.arange(len(self.X))
        idxV = find_VinXV(self.indexX, idxX, self.indexV)
        itr = 1
        while True:
            print('\niter', itr)
            print('|V|=%s |X|=%s' % (len(idxV), len(idxX)))

            fullXV = np.array([])

            for f in self.feats:
                feat = [f, ]
                XF = self.X[idxX][:, feat]
                VF = self.X[idxV][:, feat]
                idxB = Core(XF, VF, self.param, feat).idxB
                fullXV = np.append(fullXV, idxB)

            countX = np.array([len(np.where(fullXV == i)[0]) for i in range(len(idxX))]).astype(int)
            idxXwV = np.array([int(i) for i, x in enumerate(idxX) if x not in idxV])

            UPDcountX = countX[idxXwV]
            idxNewX = idxX[idxXwV]
            idxNewV = np.array([int(idx) for i, idx in enumerate(idxNewX) if UPDcountX[i] >= max(UPDcountX)]).astype(int)

            idxVit = np.union1d(idxV, idxNewV)

            # idxX = idxNewX[[int(i) for i, x in enumerate(idxNewX) if x not in idxVit]]
            if len(idxVit) > 135:
                return Result(idxV, self.param, self.imp, 'adaXallV')
            idxV = idxVit
            itr += 1

    def allVoneF(self):
        IDX = np.array([])
        for f in self.feats:
            feat = [f, ]
            X = self.X[:, feat]
            V = self.V[:, feat]
            Y = self.Y[:, feat]
            idxB = Core(X, Y, V, self.param, feat).idxB
            IDX = np.append(IDX, idxB)
        countX = np.array([len(np.where(IDX == i)[0]) for i in range(len(self.X))])
        idxB, _ = self.count_border_blade(self.param.border, countX)
        return Result(idxB, self.param, self.imp, 'allVoneF')

    def oneVallF(self):
        IDX = np.array([])
        for v in self.V:
            V = np.array([v])[:, self.feats]
            X = self.X[:, self.feats]
            Y = self.Y[:, self.feats]
            idxB = Core(X, Y, V, self.param, self.feats).idxB
            IDX = np.append(IDX, idxB)
        countX = np.array([len(np.where(IDX == i)[0]) for i in range(len(self.X))])
        idxB, _ = self.count_border_blade(self.param.border, countX)
        return Result(idxB, self.param, self.imp, 'oneVallF')

    def oneVoneP(self):
        fullXV = np.array([]).astype(int)
        for vi, v in enumerate(self.V):
            idxXVF = np.array([]).astype(int)
            for f in self.feats:
                feat = [f, ]
                XF = self.X[:, feat]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]
                idxB = Core(XF, YF, VF, self.param, feat).idxB
                idxXVF = np.append(idxXVF, idxB)

            countX = np.array([len(np.where(idxXVF == i)[0]) for i in range(len(self.X))]).astype(int)
            idxXv, _ = self.count_border_blade(self.param.border, countX)
            fullXV = np.append(fullXV, idxXv)

        idxXV = np.unique(fullXV).astype(int)
        return Result(idxXV, self.param, self.imp, 'oneVoneP')

    def oneVoneP_Y(self):
        alpha_const_Y_array = [[] for i in range(len(self.V))]
        countY_const_array = []

        for vi, v in enumerate(self.V):
            idxYVF = np.array([]).astype(int)
            for fi, f in enumerate(self.feats):
                feat = [f, ]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]
                res = Core(YF, YF, VF, self.param, feat)
                alpha_const_Y_array[vi].append(res.alpha_const)
                idxYVF = np.append(idxYVF, res.idxB)

            countY = np.array([len(np.where(idxYVF == i)[0]) for i in range(len(self.Y))]).astype(int)
            idxYv, border_const_Y = self.count_border_blade(self.param.border, countY)
            countY_const_array.append(border_const_Y)


        fullXV = np.array([]).astype(int)
        for vi, v in enumerate(self.V):
            idxXVF = np.array([]).astype(int)
            for fi, f in enumerate(self.feats):
                feat = [f, ]
                XF = self.X[:, feat]
                YF = self.Y[:, feat]
                VF = np.array([v])[:, feat]
                res = Core(XF, YF, VF, self.param, feat, alpha=alpha_const_Y_array[vi][fi])
                idxXVF = np.append(idxXVF, res.idxB)

            countX = np.array([len(np.where(idxXVF == i)[0]) for i in range(len(self.X))]).astype(int)
            idxXv = self.count_border_blade(self.param.border, countX, border_const=countY_const_array[vi])
            fullXV = np.append(fullXV, idxXv)

        idxXV = np.unique(fullXV).astype(int)
        return Result(idxXV, self.param, self.imp, 'oneVoneP_Y')




    def allVoneP_iter(self):
        idxX = np.arange(len(self.X))
        idxV = find_VinXV(self.indexX, idxX, self.indexV)
        itr = 1
        while True:
            fullIDX = np.array([])
            for f in self.feats:
                feat = [f, ]
                XF = self.X[idxX][:, feat]
                VF = self.X[idxV][:, feat]
                YF = self.Y[idxV][:, feat]
                idxB = Core(XF, YF, VF, self.param, feat).idxB
                fullIDX = np.append(fullIDX, idxB)

            countX = np.array([len(np.where(fullIDX == i)[0]) for i in range(len(idxX))]).astype(int)
            idxNewV = self.count_border_blade(self.param.border, countX)
            UPDidxV = np.union1d(idxV, idxNewV)
            if len(UPDidxV) > 140:
                return Result(idxV, self.param, self.imp, 'allVonePiter')
            idxV = UPDidxV
            itr += 1

