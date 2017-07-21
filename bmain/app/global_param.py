import numpy as np

class ParamGlobal:
    def __init__(self):
        self.zone = 'kvz'  # проверить M, шаг сетки, границы карты,
        self.gridVers = False
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

        self.border = ['h(X)', 8]




    def global_feats(self):
        FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11 old feats
        #FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17, 18]  # 14 mag
        # FEATS_GLOBAL = [1, 2, 3, 4, 24, 25, 26, 47, 48, 49] #10 new feats

        #FEATS_GLOBAL = [58, 59, 60, 61, 62, 63, 64]  #bougue
        # FEATS_GLOBAL = np.array([[6, 63, 120, 177]])
        #FEATS_GLOBAL = np.arange(1, 50)

        return FEATS_GLOBAL

    def get_field_coords(self):
        # coordinatesPoly = [30, 52, 37, 46]#kvz+crim
        coordinatesPoly = [36, 52, 37, 46] # kvz
        #coordinatesPoly = [32.5, 37.3, 43.2, 46] # crimea

        return np.array(coordinatesPoly)

    def get_squar_poly_coords(self):
        # [[32.5, 44.0], [33.7, 46.0], [40.2, 44.8], [40.2, 44], [41.5, 44], [41.7, 44.5], [43.2, 44.5],
        coordinatesPoly = [[36.8, 44.2], [37.4, 45.35], [40.2, 44.8], [40.2, 44], [41.5, 44], [41.7, 44.5],
                           [43.2, 44.5],
                           [44.1, 43.3], [47.8, 43.3], [51.2, 40.2], [51.2, 39.8], [48.5, 37.9], [47, 38.9],
                           [46.1, 38.2],
                           [44.2, 38.8], [41, 39.6], [40.4, 40.1], [41, 40.9], [41, 42.8]] #kvz
        #coordinatesPoly = [[32.7, 43.5], [32.7, 45], [37, 45.8], [37, 44.5]]

        return np.array(coordinatesPoly)