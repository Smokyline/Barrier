import numpy as np
from module.drawMap import check_pix_pers, acc_check, visuaMSdiffPix_bw, visuaMSdiffPix_color



def idx_diff_runnerAwB(A, B):
    """нахождение индексов A, которых нет в B """
    narr = np.array([]).astype(int)
    for i, a in enumerate(A):
        if a not in B:
            narr = np.append(narr, a)
    return narr


def points_diff_runnerAwB(A, B):
    """разность между A и B по координатам"""
    narrA = np.empty((0, 2))
    for i, a in enumerate(A):
        fDimEQLS = np.where(B[:, 0] == a[0])[0]
        if len(fDimEQLS) > 0:
            sDimEQLA = np.where(B[fDimEQLS, 1] == a[1])[0]
            if len(sDimEQLA) == 0:
                narrA = np.append(narrA, np.array([a]), axis=0)
        else:
            narrA = np.append(narrA, np.array([a]), axis=0)
    return narrA


class CompareAlgh:
    def __init__(self, imp, vis, barrierB, coraX):
        self.algA = barrierB
        self.algB = coraX

        self.union = np.union1d(self.algA, self.algB)
        self.inters = list(set(self.algA) & set(self.algB))
        self.AwB = idx_diff_runnerAwB(self.algA, self.algB)
        self.BwA = idx_diff_runnerAwB(self.algB, self.algA)

        self.persUnion = check_pix_pers(imp.data_coord[self.union], grid=imp.gridVers)
        self.persA = check_pix_pers(imp.data_coord[self.algA], grid=imp.gridVers)
        self.persB = check_pix_pers(imp.data_coord[self.algB], grid=imp.gridVers)

        self.accA = acc_check(imp.data_coord[self.algA], imp.eq_all, r=vis.r)
        self.accB = acc_check(imp.data_coord[self.algB], imp.eq_all, r=vis.r)
        self.accUnion = acc_check(imp.data_coord[self.union], imp.eq_all, r=vis.r)

        self.data_coord = imp.data_coord
        self.save_path = imp.save_path
        self.vis = vis

    def tanimoto(self):
        """мера Танимото (пересечение\объединение)"""
        return round(len(self.inters) / len(self.union), 2)

    def differ(self):
        """разность (объединение минус пересечение)"""
        return np.union1d(self.AwB, self.BwA)

    def visual_compare(self, EXT):
        compare_title = 'Barrier(%s%s) vs Cora(%s%s) U=%s(%s%s) ' % (
            self.persA, '%', self.persB, '%', len(self.union), self.persUnion, '%')
        compare_title2 = '%s vs %s accU=%s tanim=%s BnC=%s B/C=%s C/B=%s' % (
            self.accA, self.accB, self.accUnion, self.tanimoto(), len(self.inters), len(self.AwB), len(self.BwA))

        self.vis.node_ln_diff_res(SETS=[self.data_coord[self.inters], self.data_coord[self.AwB], self.data_coord[self.BwA]], EXT=EXT, labels=['BnC', 'B/C', 'C/B'], title=compare_title, title2=compare_title2)
        #self.vis.visual_circle(res=self.algB, title='Cora-3')

        #self.vis.bw_stere_res(B=self.data_coord[self.algA], head_title='Барьер', circle_color='#aeaeae')
        #self.vis.bw_stere_res(B=self.data_coord[self.algB], head_title='Кора-3', circle_color='none')
        #self.vis.bw_stere_res(B=self.data_coord[self.union], head_title='Объединение', circle_color='#898989')

        #visuaMSdiffPix_bw(self.data_coord[self.union], self.data_coord[self.inters], r=0.225, direc=self.save_path, title='Разность площадей')
        visuaMSdiffPix_color(self.data_coord[self.union], self.data_coord[self.inters], r=0.225, direc=self.save_path, title='Разность площадей')

