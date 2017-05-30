from main.app.barrier_mod import BarrierMod
from main.supportAlg.drawMap import Visual
from main.supportAlg.import_data import ImportData
from main.supportAlg.comparison import CompareAlgh
from main.alghTools.tools import read_cora_res

import numpy as np



class ParamGlobal:
    def __init__(self):
        self.s = -2

        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.epsilon = False
        self.beta = False
        self.mcos = False
        # self.range_const = 0.945
        self.range_const = False

        self.metrics = False
        self.delta = False
        self.manhattan = False
        self.nchCount = False

        self.vector = True

        self.border = ['h(X)', 1]

        #TODO синхронизация с param_conts_comb


    def global_feats(self):
        #FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11 old feats
        #FEATS_GLOBAL = [1, 2, 3, 4, 24, 25, 26, 47, 48, 49] #10 new feats

        #FEATS_GLOBAL = [122, 123, 124]
        #FEATS_GLOBAL = np.array([[6, 63, 120, 177]])
        FEATS_GLOBAL = np.array([[63]])

        return FEATS_GLOBAL


imp = ImportData(zone='kvz', gridVers=True)
bar = BarrierMod(imp, ParamGlobal())

r = bar.oneVoneP()
#r = bar.oneVoneP_Y()


#r = bar.simple()
#r = bar.oneVoneP()
#r = bar.allVoneF()
#r = bar.adaXoneV()
# r = bar.adaXfullV()
#r = bar.oneVoneP()


print(r.title)

imp.set_save_path(folder_name='', res=r)
vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
if not imp.gridVers:
    vis.visual_circle(res=r.result, title=r.title)
else:
    vis.grid_res(imp.data_coord[r.result], title=r.title, r=0.2)



#vis.ln_to_grid(r.result, 'barrier')

#cora_res = read_cora_res(imp.indexX, c=1)
#vis.ln_to_grid(cora_res, 'cora')
#c = CompareAlgh(imp=imp, vis=vis, barrierX=r.result, coraX=cora_res)
#c.visual_compare(result=r)