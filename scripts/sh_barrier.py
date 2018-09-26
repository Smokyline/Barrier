import numpy as np
import multiprocessing as mp

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Circle
from matplotlib.patches import RegularPolygon
import matplotlib.patches as patches

from module.import_data import ImportData
from barrier.parameters import ParamGlobal
from module.drawMap import Visual
from barrier.barrier import Core
from module.tools import *


def ploting_nodes(high_seism_nodes, miss_B_nodes, X, EXT, sample_coord, title, path):
    eq_all, eq_ist, eq_instr, eqLegend = imp.get_eq_stack()
    m_c = gp.get_field_coords()

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    m = Basemap(llcrnrlat=m_c[2], urcrnrlat=m_c[3],
                llcrnrlon=m_c[0], urcrnrlon=m_c[1],
                resolution='h')
    # m.fillcontinents(color='white', lake_color='aqua',zorder=0, alpha=.5)
    m.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True, zorder=0)

    m.drawcountries(zorder=1, linewidth=0.6)
    m.drawcoastlines(zorder=1, linewidth=0.6)
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
    meridians = np.arange(0., 360, 4)
    m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)

    if EXT is not None:
        ax.scatter(EXT[:, 0], EXT[:, 1], c='#fd41cd', marker='s', s=20, linewidths=0.0, label='e2xt')

    plt.scatter(X[:, 0], X[:, 1], c='k', marker='.', lw=0, zorder=3, s=13)
    plt.plot(ParamGlobal().get_squar_poly_coords(), c='b', alpha=0.7, zorder=2)

    for x, y, r in zip(high_seism_nodes[:, 0], high_seism_nodes[:, 1],
                       [gp.radius for i in range(len(high_seism_nodes))]):
        # круги без проекции
        # circle_B = ax.add_artist( Circle(xy=(x, y),radius=r, alpha=0.9, linewidth=0.75, zorder=2, facecolor=color, edgecolor="k"))

        # эллипсы с проекцией
        m.tissot(x, y, r, 50, alpha=0.9, linewidth=0.75, zorder=2, facecolor='b', edgecolor="k")

        plt.scatter(x, y, c='g', marker='.', lw=0, zorder=4, s=15)

    for x, y, r in zip(sample_coord[:, 0], sample_coord[:, 1], [gp.radius for i in range(len(sample_coord))]):
        # круги без проекции
        # circle_B = ax.add_artist( Circle(xy=(x, y),radius=r, alpha=0.9, linewidth=0.75, zorder=2, facecolor=color, edgecolor="k"))

        # эллипсы с проекцией
        m.tissot(x, y, r, 50, alpha=0.9, linewidth=0.75, zorder=2, facecolor='g', edgecolor="k")

        plt.scatter(x, y, c='b', marker='.', lw=0, zorder=4, s=15)

    plt.scatter(miss_B_nodes[:, 0], miss_B_nodes[:, 1], c='y', marker='X', linewidths=0.7, zorder=5, s=21, alpha=1,
                edgecolors='k')

    # исторические - треугольник инструментальные - круг
    plt.scatter(eq_ist[:, 0], eq_ist[:, 1], c='r', marker='^', linewidths=0.7, zorder=4, s=18, alpha=0.8,
                edgecolors='k')
    plt.scatter(eq_instr[:, 0], eq_instr[:, 1], c='r', marker='o', linewidths=0.75, zorder=4, s=18, alpha=0.8,
                edgecolors='k')

    # все - круги
    # plt.scatter(self.eq_all[:, 0], self.eq_all[:, 1], c='r', marker='o', linewidths=0.45, zorder=4, s=20, alpha=0.8)


    scB = plt.scatter([], [], c='g', linewidth='0.5', label='B', zorder=2)
    scH = plt.scatter([], [], c='k', linewidth='0.5', label='H', zorder=2)
    scEQis = plt.scatter([], [], c='r', marker='^', linewidth='0.5', label=eqLegend[1], zorder=2)
    scEQitr = plt.scatter([], [], c='r', linewidth='0.5', label=eqLegend[2], zorder=2)
    plt.legend(handles=[scB, scH, scEQis, scEQitr], loc=8, bbox_to_anchor=(0.5, -0.4), ncol=2)
    plt.title(title)
    plt.savefig(path + title + '.png', dpi=400)
    plt.close()

def feats_run(que, v, X, Y, feats, gp):
        idxXvF = np.array([]).astype(int)
        alpha_const_vF = []  # константы, разделяющие множество расстояний v малого f малого
        roXvF = []  # расстояния X до v малого F большое
        for fi, feat in enumerate(feats):
            XF = X[:, feat]
            YF = Y[:, feat]
            # VF = np.array([v])[:, feat]
            VF = v[feat]

            res = Core(XF, YF, VF, gp, feat)
            idxXvF = np.append(idxXvF, res.idxB)
            roXvF.append(res.XV)
            alpha_const_vF.append(res.alpha_const)

        '''вычисление кол-ва попаданий Х в вс класс по константе alpha(v, F)'''
        countX = calc_count(idxXvF, roXvF, len(X), gp.nchCount, alpha_const_vF)

        '''разделение кол-ва попаданий X по F большому в вс класс по border'''
        idxB, countX_const = count_border_blade(gp.border, countX, border_const=None)
        que.put([countX, countX_const, idxB])

def oneVoneP(V, X, Y, gp, imp):
        feats = gp.global_feats()
        V_queue = []
        for vi, v in enumerate(V):
            v_que = mp.Queue()
            p = mp.Process(target=feats_run, args=(v_que, v, X, Y, feats, gp))
            V_queue.append(v_que)
            p.start()

        V_queue = np.array([que.get() for que in V_queue])
        countX_VF = np.ravel(V_queue[:, 0])
        countX_const_arr = np.ravel(V_queue[:, 1])
        full_idxB = np.array([]).astype(int)  # высокосейсмичные индексы
        for idxxxx in V_queue[:, 2]:
            full_idxB = np.append(full_idxB, idxxxx)

        final_idxB = np.unique(full_idxB).astype(int)
        #TODO ссылку на резалт
        return Result(final_idxB, countX_VF, countX_const_arr, gp, imp, 'oneVoneP')


gp = ParamGlobal()
EXT = read_csv_pandas('/home/ivan/Documents/workspace/resources/csv/Barrier/altai/altaySayBaikal_EXT.csv')
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='altai_mk6')

idx_1905a = [15, 17, 31, 44, 45, 46, 48, 49, 74, 75, 76, 78, 79, 83, 91]
idx_1905b = [15, 17, 31, 44, 45, 48, 49, 70, 74, 75, 76, 78, 79, 83, 91]
idx_1922 = [15, 17, 44, 45, 46, 48, 49, 70, 74, 75, 76, 78, 79, 83, 91]
idx_1938 = [15, 31, 44, 45, 46, 48, 49, 70, 74, 75, 76, 78, 79, 83, 91]
idx_1900 = [15, 44, 45, 48, 49, 74, 75, 76, 78, 79, 83, 91]

date_year = ['1905a', '1905b', '1922', '1938', '1905-1038']
SAMPLE = imp.data_sample.copy()
SAMPLE_COORD = imp.sample_coord.copy()

gp.s = -1.7
gp.border = ['ro', 7]
print('s:%s q:%s' % (gp.s, gp.border))
for j, year_idx in enumerate([idx_1905a, idx_1905b, idx_1922, idx_1938, idx_1900]):
    year_sample = np.empty((0, 20))
    year_sample_coord = np.empty((0, 2))
    year_miss_sample_coord = np.empty((0, 2))

    for i, ln_idx in enumerate(np.array(SAMPLE[:, 0]).astype(int)):
        if ln_idx in year_idx:
            year_sample = np.append(year_sample, [SAMPLE[i, :]], axis=0)
            year_sample_coord = np.append(year_sample_coord, [SAMPLE_COORD[i]], axis=0)
        else:
            year_miss_sample_coord = np.append(year_miss_sample_coord, [SAMPLE_COORD[i]], axis=0)

    r = oneVoneP(year_sample, imp.data, imp.data, gp, imp)
    title = 'year:%s s:%s q:%s |B0|:%s |B|:%s' % (date_year[j], gp.s, gp.border[1], len(year_sample), r.lenB)
    ploting_nodes(imp.data_coord[r.result], year_miss_sample_coord, imp.data_coord, EXT, year_sample_coord, title,
                  imp.save_path)

