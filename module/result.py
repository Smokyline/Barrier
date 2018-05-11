from module import tools
from module.drawMap import Visual, check_pix_pers, acc_check
from comparison.comparison_two_res import CompareAlgh

class Result:
    def __init__(self, barrier, gp, imp):
        self.hs_indexes = barrier.hs_indexes  # индексы высокосейсмичных узлов
        self.lenB = len(self.hs_indexes)
        self.lenX = len(imp.data_full)
        self.lenf = len(gp.global_feats())

        self.feats_count = barrier.hs_F_count
        self.feats_top = barrier.hs_F_top

        self.pers = check_pix_pers(imp.data_coord[self.hs_indexes], grid=gp.gridVers)  # процент занимаемой прощади
        self.acc = acc_check(imp.data_coord[self.hs_indexes], imp.eq_all, r=gp.radius, grid=gp.gridVers)  # точность

        self.alg_name = 'Barrier'
        self.param_title = tools.set_title_param(vars(gp))
        self.title = '%s B=%s(%s%s) acc=%s f=%s %s' % (self.alg_name, self.lenB, self.pers, '%', self.acc,
                                                       self.lenf, self.param_title)

        imp.set_save_path(alg_name=self.alg_name, lenf=self.lenf)
        self.imp = imp
        self.hs_coord = imp.data_coord[self.hs_indexes]

        self.visual = Visual(X=imp.data_coord, imp=imp, gp=gp, path=self.imp.save_path)

    def visual_cora(self, vers=1):
        if vers == 1:
            self.cora_res = self.imp.read_cora_res(c=1)
        else:
            self.cora_res = self.imp.read_cora_res_2()

        self.visual.draw_hs_circles(res=self.cora_res, title='Cora')

    def visual_hs_feats(self):
        self.visual.feats_diagram(self.feats_count, 'count_X')
        self.visual.feats_diagram(self.feats_top, 'top_B0')

    def visual_barrier(self):

        self.visual.draw_hs_circles(res=self.hs_indexes, title=self.title)


    def compare_algh(self):
        c = CompareAlgh(imp=self.imp, vis=self.visual, barrierB=self.hs_indexes, coraX=self.cora_res)
        c.visual_compare(EXT=self.imp.EXT)

    def save_res_to_csv(self):
        one_zero_arr = []
        for i in range(self.lenX):
            if i in self.hs_indexes:
                one_zero_arr.append(1)
            else:
                one_zero_arr.append(0)
        tools.save_res_idx_to_csv(one_zero_arr, self.title, self.imp.save_path)
        tools.save_res_coord_to_csv(self.hs_coord, self.title, self.imp.save_path)