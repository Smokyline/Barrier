from main.app.barrier_mod import BarrierMod
from main.supportAlg.drawMap import Visual
from main.supportAlg.import_data import ImportData
import numpy as np


class ParamGlobal:
    def __init__(self):
        self.s = False

        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.epsilon = False
        self.beta = False
        self.mcos = False
        #self.range_const = 0.945
        self.range_const = 0.945

        self.metrics = False
        self.delta = False
        self.vector = False
        self.manhattan = True

        self.nchCount = False

        self.border = ['h(X)', 1]
        #self.border = ['ro', 18]
        #self.border = ['kmeans', 25]
        #self.border = ['pers', 12]

        #self.FEATS_GLOBAL = [56, 57, 58]
        #self.FEATS_GLOBAL = np.array([[56, 57, 58]])
        #self.FEATS_GLOBAL = np.array([[64,65,66]])
        #self.FEATS_GLOBAL = np.array([[111, 112, 113]])
        self.FEATS_GLOBAL = np.array([[162, 163, 164, 165, 166, 167, 168, 169, 170, 171]])

        # self.FEATS_GLOBAL = np.array([[8, 9, 10], [31, 32, 33], [54, 55, 56], [77, 78, 79]])
        #self.FEATS_GLOBAL = np.array([[31, 32, 33]])


    def global_feats(self):

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


        for s in np.arange(-2.0, -2.1, -0.2):
            #gp.s = s
            gp.s = False

            res = run_bar(imp)
            res_idx = res.result


            imp.set_save_path(folder_name='', res=res)
            vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
            if gridVers:
                print(res.title)
                vis.grid_res(imp.data_coord[res_idx], title=res.title, r=0.2)
            else:
                print(res.title)
                vis.visual_circle(res=imp.data_coord[res_idx], title=res.title)



gp = ParamGlobal()

comb_grid_param()
#run_union_bar(gridVers=True)

"""
time_start = int(round(time.time() * 1000))
                print( int(round(time.time() * 1000)) - time_start)  # millsec

"""