from main.app.barrier_class import BarrierMod
from main.supportAlg.import_data import ImportData
from main.alghTools.tools import res_to_txt
import numpy as np


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
        # self.border = ['h(X)', 2.6]
        self.border = ['ro', 8]
        # self.border = ['kmeans', 2]
        # self.border = ['pers', 30]

        self.FEATS_GLOBAL = None
        # self.FEATS_GLOBAL = np.array([[20, 21, 22]])

    def global_feats(self):
        # self.FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
        # FEATS_GLOBAL = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])

        return self.FEATS_GLOBAL


def run_bar():
    bar = BarrierMod(imp, gp)
    alg_stack = {
        # 'oneVoneP': bar.oneVoneP,
        'oneVoneP_Y': bar.oneVoneP_Y
    }
    for name, algmod in alg_stack.items():
        res = algmod()
        print(res.title)

        # algname, кол-во объектов, точность, % территории, параметры
        alg_title = res.alg_name + '_' + str(gp.FEATS_GLOBAL)
        row = [alg_title.replace(" ", "_"), res.lenB, res.acc, res.pers, res.param_title.replace(" ", "_")[:-1]]
        imp.set_save_path(folder_name='comp', res=res)
        res_to_txt(imp.save_path + name + '.txt', row)


def param_comb():
    for s in np.arange(-1, -4.1, -0.1):
        gp.s = s
        for h in np.arange(7, 12.2, 1):
            gp.border = ['ro', h]
            run_bar()


gp = ParamGlobal()
gp.FEATS_GLOBAL = [8, 9, 10]
imp = ImportData(gridVers=True)

param_comb()
