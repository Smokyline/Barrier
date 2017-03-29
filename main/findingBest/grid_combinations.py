from main.app.barrier_class import BarrierMod
from main.alghTools.drawMap import Visual
from main.alghTools.import_data import ImportData
from main.alghTools.tools import res_to_txt, set_title_param
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
        self.bar = False
        self.metrix = False
        self.nchCount = True

        # self.border = False
        #self.border = ['h(X)', 1]
        self.border = ['ro', 6.8]
        #self.border = ['kmeans', 25]
        # self.border = ['pers', 30]

        self.FEATS_GLOBAL = None
        # self.FEATS_GLOBAL = np.array([[20, 21, 22]])

    def global_feats(self):
        # self.FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
        # FEATS_GLOBAL = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])

        return self.FEATS_GLOBAL


def change_gp(gp, s, border, feat):
    gp.s = s
    gp.border = border
    gp.FEATS_GLOBAL = feat

def run_union_bar(gridVers=False):
    imp = ImportData(zone='kvz', gridVers=gridVers)

    prm_set = (
        (-3.1, ['ro', 11], [8, 9, 10]),
        (-2.5, ['ro', 12], [31, 32, 33]),
        (-4, ['ro', 12], [54, 55, 56]),
        (-3.9, ['ro', 7], [77, 78, 79]),

    )

    RE_count = []
    RE_mean_const = []
    for prm in prm_set:
        change_gp(gp, prm[0], prm[1], prm[2])
        res = run_bar(imp)

        print(res.title)
        RE_count.append(res.countX)
        RE_mean_const.append(res.psi_mean_const)
    RE_count = np.array(RE_count)  # (4, 16, 3934)
    RE_mean_const = np.array(RE_mean_const)  # (4, 16, 3934)

    bar = BarrierMod(imp, gp)

    ###
    final_res = bar.union_nch_count_res(RE_count, RE_mean_const, gp, imp, theta=['ro', 18])
    print(final_res.title)
    imp.set_save_path(folder_name='nch_sum', res=final_res)
    vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
    vis.grid_res(imp.data_coord[final_res.result], title=final_res.title, r=0.2)
    ###


def run_bar(imp):
    bar = BarrierMod(imp, gp)

    res = bar.oneVoneP()
    #res = bar.oneVoneP_Y()

    return res


def comb_grid_param():
    for gridVers in [True]:
        imp = ImportData(zone='kvz', gridVers=gridVers)
        gp.FEATS_GLOBAL = [31, 32, 33]
        # gp.FEATS_GLOBAL = [76, 77, 78, 79, 80, 81, 82]
        # gp.FEATS_GLOBAL = np.array([[8, 9, 10], [31, 32, 33], [54, 55, 56], [77, 78, 79]])
        #gp.FEATS_GLOBAL = np.array([[31, 32, 33]])
        # gp.FEATS_GLOBAL = np.array([[83, 84, 85]])
        # gp.FEATS_GLOBAL = [n for n in range(1, len(imp.data_full[0]))]

        # print([imp.col[i] for i in gp.FEATS_GLOBAL])

        for s in np.arange(-2.5, -4.1, -0.2):
            gp.s = s

            res = run_bar(imp)
            res_idx = res.result

            imp.set_save_path(folder_name='', res=res)
            vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
            if gridVers:
                # acc_pers, acc_count, miss_count = calc_acc_pixpoly(imp.data_coord[res_idx], imp.eq_all, delta=0.2)
                # +str(gp.FEATS_GLOBAL)
                # title = '%s B=%s(%s%s) acc=%s(%s%s) f=%s %s' % (res.alg_name, len(res_idx), res.pers, '%', res.acc, round(acc_pers, 3), '%',
                # res.lenf, res.param_title)
                print(res.title)
                vis.grid_res(imp.data_coord[res_idx], title=res.title, r=0.2)
            else:
                print(res.title)
                vis.color_res(res=imp.data_coord[res_idx], title=res.title)



gp = ParamGlobal()

comb_grid_param()
#run_union_bar(gridVers=True)
