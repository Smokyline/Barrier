from main.app.barrier_class import BarrierMod
from main.alghTools.drawMap import Visual
from main.alghTools.import_data import ImportData
from main.alghTools.tools import res_to_txt, set_title_param
import numpy as np
import time


class ParamGlobal:
    def __init__(self):
        self.q = -1
        self.s = -0.7  # AlphaMax!!
        self.r = False
        self.delta = False
        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.epsilon = False
        self.bar = False
        self.border = False

    def global_feats(self):
        FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
        #FEATS_GLOBAL = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])

        return FEATS_GLOBAL


def run_bar(name_mod, alg_mod):

    res = alg_mod()

    if res.pers == 0:
        psi = 0
    else:
        #a*(1 - s/100)
        psi = res.acc * (1 - res.pers/100)

    # algname, кол-во объектов, точность, % территории, параметры
    row = [res.alg_name, res.lenB, res.acc, res.pers, round(psi, 4), res.param_title.replace(" ", "_")[:-1]]
    res_to_txt(imp.save_path + name_mod +'.txt', row)

    return res


gp = ParamGlobal()
imp = ImportData(folder_name='comp')
bar = BarrierMod(imp, gp)

alg_stack = {
    'simple': bar.simple,
    'iter_learn': bar.iter_learning,
    'adaXoneV': bar.adaXoneV,
    'adaXallV': bar.adaXallV,
    'allVoneF': bar.allVoneF,
    'oneVallF': bar.oneVallF,
    'oneVoneP': bar.oneVoneP,
    'allVoneP_iter': bar.allVoneP_iter,
}



for name in ['allVoneF', 'oneVallF', 'iter_learn', 'allVoneP_iter','adaXallV']:

    algmod = alg_stack[name]
    start_time = int(time.time() * 1000)


    for s in np.arange(-0.1, -1.7, -0.2):
        gp.s = s
        for q in np.arange(-0.5, -1.7, -0.2):
            gp.q = q
            for delta in [False, True]:
                gp.delta = delta
                for alphaMax in [False, True]:
                    gp.alphaMax = alphaMax

                    if name in ['simple', 'iter_learn', 'adaXoneV', 'adaXallV']:
                        try:
                            result = run_bar(name, algmod)
                            print(result.title)
                        except Exception as e:
                            print(e)
                    else:
                        for border in [['h(X)', 6], ['h(X)', 7], ['h(X)', 8], ['h(X)', 9], ['h(X)', 10],
                                       ['pers', 5], ['pers', 10], ['pers', 15], ['pers', 20], ['pers', 25], ['pers', 30],
                                       ['kmeans', 2], ['kmeans', 3], ['kmeans', 4], ['kmeans', 5], ['kmeans', 6], ['kmeans', 7],
                                       ['kmeans', 8], ['kmeans', 9], ['kmeans', 10]]:
                            gp.border = border
                            try:
                                result = run_bar(name, algmod)
                                print(result.title)
                            except Exception as e:
                                print(e)




    total_time = int(time.time() * 1000) - start_time  # ms
    H, M, S = total_time / 1000 / 60 / 60, total_time / 1000 / 60, total_time / 1000
    print('\n H:%i M:%i S:%i' % (H, M, S,))