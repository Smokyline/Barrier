from testing.app.barrier_class import BarrierMod
from testing.alghTools.drawMap import Visual, calc_acc_pixpoly
from testing.alghTools.import_data import ImportData
from testing.alghTools.tools import res_to_txt, set_title_param
import numpy as np
import time


class ParamGlobal:
    def __init__(self):
        self.q = -1
        self.s = -0.7  # AlphaMax!!
        self.delta = False
        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.epsilon = False
        self.bar = False
        #self.border = False
        #self.border = ['h(X)', 5]
        self.border = ['ro', 8]
        #self.border = ['kmeans', 10]

        self.FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11


    def global_feats(self):
        # FEATS_GLOBAL = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])

        return self.FEATS_GLOBAL


def run_bar():
    bar = BarrierMod(imp, gp)

    #res = bar.simple()
    res = bar.oneVoneP()
    #res = bar.oneVallF()

    if res.pers == 0:
        psi = 0
    else:
        # a*(1 - s/100)
        psi = res.acc * (1 - res.pers / 100)


    return res



gp = ParamGlobal()


for gridVers in [True]:
    imp = ImportData(zone='kvz', gridVers=gridVers)

    #if gridVers:
    #    gp.FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #else:
    #    gp.FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]

    gp.FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for s in np.arange(-2.4, -4.5, -0.2):
        gp.s = s
        for q in np.arange(-0.1, -0.15, -0.2):
            gp.q = q

            res = run_bar()

            imp.set_save_path(folder_name='', res=res)
            vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)

            if gridVers:
                acc_pers, acc_count, miss_count = calc_acc_pixpoly(imp.data_coord[res.result], imp.eq_all, delta=0.15)
                title = '%s B=%s(%s%s) acc=%s(%s%s) f=%s %s' % (res.alg_name, res.lenB, res.pers, '%', acc_count, round(acc_pers, 3), '%',
                                                                                        res.lenf, res.param_title)
                print(title)
                vis.grid_res(imp.data_coord[res.result], title=title)
            else:
                print(res.title)
                vis.color_res(B=imp.data_coord[res.result], title=res.title, V=imp.data_coord[res.V])




















#run_comb()