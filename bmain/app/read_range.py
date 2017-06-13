from bmain.supportAlg.import_data import ImportData

from bmain.alghTools.supportAlg.drawMap import Visual
from bmain.alghTools.tools import *
from bmain.barrier.barrier_version import Result


def read_csv(path):
    """чтение csv файла по col колонкам"""
    frame = pd.read_csv(path, header=None, sep=';', decimal=",")
    array = np.array(frame.values)
    return array


class ParamGlobal:
    def __init__(self):
        self.s = False

        self.kmeans = False
        self.alphaMax = False
        self.pers = False
        self.epsilon = False
        self.beta = False
        self.mcos = 0.999999

        self.metrics = True
        self.delta = False
        self.vector = False

        self.nchCount = True

        #self.border = ['h(X)', 2.5]
        #self.border = ['ro', 18]
        #self.border = ['kmeans', 25]
        self.border = ['pers', 3]

        #self.FEATS_GLOBAL = None
        self.FEATS_GLOBAL = [31, 32, 33]
        #self.FEATS_GLOBAL = [37, 38, 39, 40, 41]
        #self.FEATS_GLOBAL = np.array([[37, 38, 39, 40, 41]])

        # self.FEATS_GLOBAL = [76, 77, 78, 79, 80, 81, 82]
        # self.FEATS_GLOBAL = np.array([[8, 9, 10], [31, 32, 33], [54, 55, 56], [77, 78, 79]])
        #self.FEATS_GLOBAL = np.array([[31, 32, 33]])

        # print([imp.col[i] for i in gp.FEATS_GLOBAL])
        # self.FEATS_GLOBAL = np.array([[20, 21, 22]])

    def global_feats(self):
        # self.FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
        # FEATS_GLOBAL = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])

        return self.FEATS_GLOBAL



imp = ImportData(zone='kvz', gridVers=True)
gp = ParamGlobal()


COUNT_V = 17
COUNT_F = len(gp.FEATS_GLOBAL)

path = '/Users/Ivan/Documents/workspace/result/Barrier/XVrange/'

full_idxB = np.array([]).astype(int)  # высокосейсмичные индексы
countX_VF = []  # кол-во попаданий X в вс класс при v малом по признакам
countX_const_arr = []  # константы границ, разделяющие кол-во попаданий по признакам

for vi in range(COUNT_V):
    idxXvF = np.array([]).astype(int)
    roXvF = []  # расстояния X до v малого F большое
    alpha_const_vF = []  # константы, разделяющие множество расстояний v малого f малого

    for fi in range(COUNT_F):
        ro_Xv = read_csv(path + 'XVrange%i-%i.csv' % (vi+1, fi+1)).T[0]
        idxB, alpha_const = found_nch_param_border(np.abs(ro_Xv - 1), gp.beta, gp.mcos)
        roXvF.append(ro_Xv)
        alpha_const_vF.append(alpha_const)
    countX = calc_count(idxXvF, roXvF, len(roXvF[0]), nch=True, alpha_const_vF=alpha_const_vF)

    countX_VF.append(countX)

    '''разделение кол-ва попаданий X по F большому в вс класс по border'''
    idxB, countX_const = count_border_blade(gp.border, countX, border_const=None)
    countX_const_arr.append(countX_const)
    print('\nlen v%iП: %i' % (vi, len(idxB)), '   alpha', alpha_const_vF, end='\n----------------------------\n')


    full_idxB = np.append(full_idxB, idxB)

final_idxB = np.unique(full_idxB).astype(int)
res = Result(final_idxB, countX_VF, countX_const_arr, gp, imp, 'oneVoneP')
res_idx = res.result

imp.set_save_path(folder_name='', res=res)
vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
print(res.title)
vis.grid_res(imp.data_coord[res_idx], title=res.title, r=0.2)
