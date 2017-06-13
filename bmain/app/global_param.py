import numpy as np

class ParamGlobal:
    def __init__(self):
        self.zone = 'kvz'
        self.gridVers = True
        self.ln_field = False
        self.s = -0.7

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

        self.vector = False

        self.border = ['h(X)', 25]




    def global_feats(self):
        #FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11 old feats
        # FEATS_GLOBAL = [1, 2, 3, 4, 24, 25, 26, 47, 48, 49] #10 new feats

        #FEATS_GLOBAL = [58, 59, 60, 61, 62, 63, 64]  #bougue
        # FEATS_GLOBAL = np.array([[6, 63, 120, 177]])
        FEATS_GLOBAL = np.arange(1, 50)

        return FEATS_GLOBAL
