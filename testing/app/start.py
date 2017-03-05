from testing.app.barrier_class import BarrierMod
from testing.alghTools.drawMap import Visual
from testing.alghTools.import_data import ImportData



class ParamGlobal:
    def __init__(self):
        self.q = -1.1
        self.s = -1.4  # AlphaMax!!
        self.delta = False
        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.bar = False
        #self.border = False
        #self.border = ['h(X)', 7]
        #self.border = ['kmeans', 10]
        self.border = ['ro', 8]

    def global_feats(self):
        #FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
        # FEATS_GLOBAL = [1, 2, 3, 4, 5, 13, 14, 15] #8
        # FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17, 18] #14
        # FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] #18
        # FEATS = [1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]
        #FEATS_GLOBAL = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]

        #FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  #grid
        FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  #grid

        return FEATS_GLOBAL

imp = ImportData(zone='kvz', gridVers=True)
bar = BarrierMod(imp, ParamGlobal())

#r = bar.oneVoneP()
#r = bar.simple()
#r = bar.oneVoneP()
r = bar.oneVoneP_Y()
#r = bar.allVoneF()
#r = bar.adaXoneV()
# r = bar.adaXfullV()
#r = bar.oneVoneP()


print(r.title)

imp.set_save_path(folder_name='', res=r)
vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
vis.color_res(B=imp.data_coord[r.result], title=r.title, V=imp.data_coord[r.V])



#c = CompareAlgh(barrierX=r.result, coraX=read_cora_res(idxCX, c=1))
