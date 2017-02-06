from testing.app.barrier_class import BarrierMod
from testing.alghTools.drawMap import Visual
from testing.alghTools.import_data import ImportData
import numpy as np


class ParamGlobal:
    def __init__(self):
        self.q = -0.9
        self.s = -0.1  # AlphaMax!!
        self.r = False
        self.delta = False
        self.kmeans = False
        self.alphaMax = True
        self.pers = False
        self.bar = False
        #self.border = False
        #self.border = ['h(X)', 6]
        self.border = ['kmeans', 2]

    def global_feats(self):
        FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
        # FEATS_GLOBAL = [1, 2, 3, 4, 5, 13, 14, 15] #8
        # FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17, 18] #14
        # FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] #18
        # FEATS = [1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]
        #FEATS_GLOBAL = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]

        return FEATS_GLOBAL

imp = ImportData(folder_name='test')
bar = BarrierMod(imp, ParamGlobal())

#r = bar.oneVoneP()
#r = bar.simple()
r = bar.allVoneF()
# r = bar.adaXoneV()
# r = bar.adaXfullV()


print(r.title)

vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
vis.color_res(B=imp.data_coord[r.result], title=r.title, V=imp.data_coord[r.V])



#c = CompareAlgh(barrierX=r.result, coraX=read_cora_res(idxCX, c=1))
