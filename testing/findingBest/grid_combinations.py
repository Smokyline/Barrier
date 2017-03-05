from testing.app.barrier_class import BarrierMod
from testing.alghTools.drawMap import Visual, calc_acc_pixpoly
from testing.alghTools.import_data import ImportData
from testing.alghTools.tools import res_to_txt, set_title_param
import numpy as np
import itertools
import time


class ParamGlobal:
    def __init__(self):
        self.q = False
        self.s = -1  # AlphaMax!!
        self.delta = False
        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.epsilon = False
        self.bar = False
        self.metrix = True

        #self.border = False
        self.border = ['h(X)', 3]
        #self.border = ['ro', 10]
        #self.border = ['kmeans', 10]
        #self.border = ['pers', 50]

        self.FEATS_GLOBAL = None
        #self.FEATS_GLOBAL = np.array([[20, 21, 22]])



    def global_feats(self):
        #self.FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
        # FEATS_GLOBAL = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])

        return self.FEATS_GLOBAL


def run_bar():
    bar = BarrierMod(imp, gp)

    #res = bar.simple()
    #res = bar.oneVoneP()
    #res = bar.allVoneF()
    res = bar.oneVoneP_Y()
    #res = bar.oneVallF()

    if res.pers == 0:
        psi = 0
    else:
        # a*(1 - s/100)
        psi = res.acc * (1 - res.pers / 100)


    return res



gp = ParamGlobal()

union = False
for gridVers in [True]:
    imp = ImportData(zone='kvz', gridVers=gridVers)

    for s in np.arange(-1, -3.0, -0.2):
        gp.s = s
        #gp.FEATS_GLOBAL = [36, 37, 38, 39]
        gp.FEATS_GLOBAL = [8, 9, 10]
        #gp.FEATS_GLOBAL = np.array([[8, 9, 10], [33, 34, 35]])


        if union:
            res_idx = np.array([]).astype(int)
            for feats in [np.array([[23, 24, 25]]), np.array([[26, 27, 28]])]:
                gp.FEATS_GLOBAL = feats
                res = run_bar()
                res_idx = np.append(res_idx, res.result)
            res_idx = np.unique(res_idx)
        else:
            res = run_bar()
            res_idx = res.result



        imp.set_save_path(folder_name='', res=res)
        vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
        if gridVers:
            acc_pers, acc_count, miss_count = calc_acc_pixpoly(imp.data_coord[res_idx], imp.eq_all, delta=0.2)
            title = '%s B=%s(%s%s) acc=%s(%s%s) f=%s %s' % (res.alg_name+str(gp.FEATS_GLOBAL), len(res_idx), res.pers, '%', acc_count, round(acc_pers, 3), '%',
                                                                                    res.lenf, res.param_title)
            print(title)
            vis.grid_res(imp.data_coord[res_idx], title=title, r=0.2)
        else:
            print(res.title)
            vis.color_res(B=imp.data_coord[res_idx], title=res.title, V=None)




















#run_comb()