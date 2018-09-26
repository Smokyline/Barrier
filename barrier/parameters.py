import numpy as np

class ParamGlobal:
    def __init__(self):
        self.zone = 'kvz'  # проверить M, шаг сетки, границы карты
        self.radius = 0.225
        self.gridVers = False
        self.ln_field = False
        self.s = -1.25

        self.omega = 3



        #self.border = ['h(X)', 10]
        self.border = ['ro', 7]




    def global_feats(self):
        """
        Hmax  Hmin  DH  Top  Q  HR  Nl  Rint  DH/l  Nlc	 R1  R2  Bmax  Bmin  DB  Mmax  Mmin  DM
        0     1     2   3    4  5   6   7     8     9    10  11  12    13    14  15    16    17

        """


        #FEATS_GLOBAL = [0, 1, 2, 3, 4, 6, 9, 11, 12, 13, 14]  # 11 old feats
        FEATS_GLOBAL = [0, 1, 2, 3, 4, 6, 9, 11, 12, 13, 14, 15, 16, 17]  # 14 +mag


        #FEATS_GLOBAL = np.arange(0, 18)

        return FEATS_GLOBAL

    def get_sample_ln_idx(self):
        if self.zone == 'kvz':
            sample_ln_idx = [30, 63, 66, 81, 84, 101, 109, 174, 178, 193, 200, 218, 248, 251, 254, 288]  # kvz
        elif self.zone == 'altai':
            sample_ln_idx = [15, 17, 31, 44, 45, 46, 48, 49, 70, 74, 75, 76, 78, 79, 83, 91] # altay
        return sample_ln_idx

    def get_field_coords(self):
        if self.zone == 'kvz':
            coordinatesPoly = [36, 52, 37, 46]  # kvz
        elif self.zone == 'altai':
            coordinatesPoly = [82, 112, 45, 57] # altay
        # coordinatesPoly = [30, 52, 37, 46]#kvz+crim
        return np.array(coordinatesPoly)

    def get_squar_poly_coords(self):
        if self.zone == 'kvz':
            coordinatesPoly = [[36.8, 44.2], [37.4, 45.35], [40.2, 44.8], [40.2, 44], [41.5, 44], [41.7, 44.5],
                           [43.2, 44.5],
                           [44.1, 43.3], [47.8, 43.3], [51.2, 40.2], [51.2, 39.8], [48.5, 37.9], [47, 38.9],
                           [46.1, 38.2],
                           [44.2, 38.8], [41, 39.6], [40.4, 40.1], [41, 40.9], [41, 42.8]] #kvz"""

        elif self.zone == 'altai':
            coordinatesPoly = [[84, 48], [82, 50], [84, 53], [90, 53], [91, 56], [100, 56], [104, 53],
                           [110, 56], [110, 53], [106, 49], [92, 49] ] #altai

        return np.array(coordinatesPoly)