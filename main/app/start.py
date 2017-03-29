from main.app.barrier_class import BarrierMod
from main.supportAlg.drawMap import Visual
from main.supportAlg.import_data import ImportData
from main.supportAlg.comparison import CompareAlgh
from main.alghTools.tools import read_cora_res



class ParamGlobal:
    def __init__(self):
        self.q = False
        self.s = -0.7
        self.delta = False
        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.epsilon = False
        self.bar = False
        self.metrix = False
        self.nchCount = False

        # self.border = False
        self.border = ['h(X)', 8]
        # self.border = ['ro', 6]
        # self.border = ['kmeans', 10]
        # self.border = ['pers', 30]


    def global_feats(self):
        FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11 old feats
        #FEATS_GLOBAL = [1, 2, 3, 4, 24, 25, 26, 47, 48, 49] #10 new feats

        # FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17, 18] #14
        # FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] #18
        # FEATS = [1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]
        #FEATS_GLOBAL = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]

        #FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  #grid
        #FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  #grid

        return FEATS_GLOBAL

imp = ImportData(zone='kvz', gridVers=False)
bar = BarrierMod(imp, ParamGlobal())

#r = bar.oneVoneP()
r = bar.oneVoneP_Y()


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



vis.ln_to_grid(r.result, 'barrier')

#cora_res = read_cora_res(imp.indexX, c=1)
#vis.ln_to_grid(cora_res, 'cora')
#c = CompareAlgh(imp=imp, vis=vis, barrierX=r.result, coraX=cora_res)
#c.visual_compare(result=r)