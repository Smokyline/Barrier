from bmain.alghTools.supportAlg.drawMap import check_pix_pers, acc_check
from bmain.alghTools.tools import set_title_param, save_res_idx_to_csv


class Result:
    def __init__(self, idxB, countX, countX_const_arr, gp, imp, alg_name):
        self.result = idxB  # индексы высокосейсмичных узлов
        self.lenB = len(idxB)
        self.lenf = len(gp.global_feats())

        self.countX = countX  # кол-во попаданий по признакам
        self.psi_mean_const = countX_const_arr

        self.pers = check_pix_pers(imp.data_coord[self.result], grid=gp.gridVers)  # процент занимаемой прощади
        self.acc = acc_check(imp.data_coord[self.result], imp.eq_all, grid=gp.gridVers)  # точность

        self.alg_name = alg_name
        self.param_title = set_title_param(vars(gp))
        self.title = '%s B=%s(%s%s) acc=%s f=%s %s' % (self.alg_name, self.lenB, self.pers, '%', self.acc,
                                                       self.lenf, self.param_title)

        imp.set_save_path(alg_name=alg_name, lenf=self.lenf)
        save_res_idx_to_csv(imp.data_full, idxB, self.title, imp.save_path)

