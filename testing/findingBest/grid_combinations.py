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
        self.s = -1
        self.delta = False
        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.epsilon = False
        self.bar = True
        self.metrix = False

        #self.border = False
        self.border = ['h(X)', 3]
        #self.border = ['ro', 6]
        #self.border = ['kmeans', 10]
        #self.border = ['pers', 30]

        self.FEATS_GLOBAL = None
        #self.FEATS_GLOBAL = np.array([[20, 21, 22]])



    def global_feats(self):
        #self.FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
        # FEATS_GLOBAL = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])

        return self.FEATS_GLOBAL


def run_bar():
    bar = BarrierMod(imp, gp)

    #res = bar.simple()
    res = bar.oneVoneP()
    #res = bar.allVoneF()
    #res = bar.oneVoneP_Y()
    #res = bar.oneVallF()

    #psi = res.acc * (1 - res.pers / 100)

    return res

gp = ParamGlobal()

union = False
for gridVers in [True]:
    imp = ImportData(zone='kvz', gridVers=gridVers)
    #gp.FEATS_GLOBAL = [75]
    #gp.FEATS_GLOBAL = [76, 77, 78, 79, 80, 81, 82]
    gp.FEATS_GLOBAL = np.array([[8, 9, 10], [31, 32, 33], [54, 55, 56], [77, 78, 79]])
    #gp.FEATS_GLOBAL = np.array([[34, 35, 36]])
    #gp.FEATS_GLOBAL = np.array([[83, 84, 85]])
    #gp.FEATS_GLOBAL = [n for n in range(1, len(imp.data_full[0]))]

    #print([imp.col[i] for i in gp.FEATS_GLOBAL])

    for s in np.arange(-0.7, -6.7, -0.3):
        gp.s = s

        if union:
            res_idx = np.array([]).astype(int)
            for feats in [[33, 34, 35], [83, 84, 85]]:
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
            #+str(gp.FEATS_GLOBAL)
            title = '%s B=%s(%s%s) acc=%s(%s%s) f=%s %s' % (res.alg_name, len(res_idx), res.pers, '%', acc_count, round(acc_pers, 3), '%',
                                                                                    res.lenf, res.param_title)
            print(title)
            vis.grid_res(imp.data_coord[res_idx], title=title, r=0.2)
        else:
            print(res.title)
            vis.color_res(res=imp.data_coord[res_idx], title=res.title)


#run_comb()