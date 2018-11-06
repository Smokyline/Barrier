import math
import numpy as np
import pandas as pd


from barrier.parameters import ParamGlobal


import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Circle
from matplotlib.patches import RegularPolygon
import matplotlib.patches as patches

from PIL import Image
# from skimage.color import rgb2gray
from  scipy.misc import imsave
import os


class Visual:
    def __init__(self, X,imp, gp, path):
        self.X = X
        self.imp = imp
        self.gp = gp
        self.path = path
        self.m_c = gp.get_field_coords()
        self.pol = gp.get_squar_poly_coords()
        self.r = gp.radius
        self.eq_all, self.eq_ist, self.eq_instr, self.eqLegend = imp.get_eq_stack()
        self.train_coord = imp.train_coord
        self.data_coord = self.imp.data_coord

    def draw_hs_circles(self, res, title):
        """отображение результатов в виде кругов"""
        B = self.data_coord[res]
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')


        pol = self.pol
        m = Basemap(llcrnrlat=self.m_c[2], urcrnrlat=self.m_c[3],
                    llcrnrlon=self.m_c[0], urcrnrlon=self.m_c[1],
                    resolution='h')
        #m.fillcontinents(color='white', lake_color='aqua',zorder=0, alpha=.5)
        m.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True, zorder=0)

        m.drawcountries(zorder=1, linewidth=0.6)
        m.drawcoastlines(zorder=1, linewidth=0.6)
        parallels = np.arange(0., 90, 2)
        m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
        meridians = np.arange(0., 360, 4)
        m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)

        ax.add_patch(patches.Polygon(pol, edgecolor="b", facecolor='none', alpha=0.6, zorder=0, ))

        if self.imp.EXT is not None:
            ax.scatter(self.imp.EXT[:, 0], self.imp.EXT[:, 1], c='#fd41cd', marker='s', s=20, linewidths=0.0, label='e2xt')

        plt.scatter(self.X[:, 0], self.X[:, 1], c='k', marker='.', lw=0, zorder=3, s=13)
        plt.plot(ParamGlobal().get_squar_poly_coords(), c='b', alpha=0.7, zorder=2)

        for x, y, r in zip(B[:, 0], B[:, 1], [self.r for i in range(len(B))]):
            # круги без проекции
            #circle_B = ax.add_artist( Circle(xy=(x, y),radius=r, alpha=0.9, linewidth=0.75, zorder=2, facecolor=color, edgecolor="k"))

            #эллипсы с проекцией
            m.tissot(x, y, r, 50, alpha=0.9, linewidth=0.75, zorder=2, facecolor='b', edgecolor="k")

            plt.scatter(x, y, c='g', marker='.', lw=0, zorder=4, s=15)

        try:
            for x, y, r in zip(self.train_coord[:, 0], self.train_coord[:, 1],
                               [self.r for i in range(len(self.train_coord))]):
                # круги без проекции
                # circle_B = ax.add_artist( Circle(xy=(x, y),radius=r, alpha=0.9, linewidth=0.75, zorder=2, facecolor=color, edgecolor="k"))

                # эллипсы с проекцией
                m.tissot(x, y, r, 50, alpha=0.9, linewidth=0.75, zorder=2, facecolor='b', edgecolor="k")

                plt.scatter(x, y, c='b', marker='.', lw=0, zorder=4, s=15)
        except:
            print('NO SAMPLE COORD... SKIP')

        # исторические - треугольник инструментальные - круг
        plt.scatter(self.eq_ist[:, 0], self.eq_ist[:, 1], c='r', marker='o', linewidths=0.7, zorder=4, s=18, alpha=0.8, edgecolors='k')
        plt.scatter(self.eq_instr[:, 0], self.eq_instr[:, 1], c='r', marker='o', linewidths=0.75, zorder=4, s=18, alpha=0.8, edgecolors='k')

        # все - круги
        #plt.scatter(self.eq_all[:, 0], self.eq_all[:, 1], c='r', marker='o', linewidths=0.45, zorder=4, s=20, alpha=0.8)


        scB = plt.scatter([], [], c='g', linewidth='0.5', label='B', zorder=2)
        scH = plt.scatter([], [], c='k', linewidth='0.5', label='H', zorder=2)
        scEQis = plt.scatter([], [], c='r', marker='^', linewidth='0.5', label=self.eqLegend[1], zorder=2)
        scEQitr = plt.scatter([], [], c='r', linewidth='0.5', label=self.eqLegend[2], zorder=2)
        plt.legend(handles=[scB, scH, scEQis, scEQitr], loc=8, bbox_to_anchor=(0.5, -0.4), ncol=2)
        plt.title(title)
        plt.savefig(self.path + title + '.png', dpi=400)
        plt.close()

    def bw_stere_res(self, B, head_title, circle_color):
        """черно-белое отображение результатов в сферической проекции"""
        m = Basemap(llcrnrlat=self.m_c[2], urcrnrlat=self.m_c[3],
                    llcrnrlon=self.m_c[0], urcrnrlon=self.m_c[1],
                    resolution='l', projection='stere', lon_0=np.mean(self.m_c[:2]), lat_0=np.mean(self.m_c[2:]))

        m.fillcontinents(color='#f0f0f0', zorder=0)
        m.drawcountries(zorder=1, linewidth=0.3)
        m.drawcoastlines(zorder=1, linewidth=0.3)
        m.drawcountries(zorder=1, linewidth=0.3)
        parallels = np.arange(0., 90, 1)
        m.drawparallels(parallels, labels=[1, 0, 0, 0], zorder=1, linewidth=0.4, alpha=0.7)
        meridians = np.arange(0., 360, 2)
        m.drawmeridians(meridians, labels=[0, 0, 0, 1], zorder=1, linewidth=0.4, alpha=0.7)
        ax = plt.gca()

        X_x, X_y = m(self.X[:, 0], self.X[:, 1])
        eq_ist_x, eq_ist_y = m(self.eq_ist[:, 0], self.eq_ist[:, 1])
        eq_instr_x, eq_instr_y = m(self.eq_instr[:, 0], self.eq_instr[:, 1])

        ax.scatter(X_x, X_y, marker='.', s=11, c='k', linewidth=0, zorder=2)
        for x, y, r in zip(B[:, 0], B[:, 1], [self.r for i in range(len(B))]):
            m.tissot(x, y, r, 50, ax=ax, alpha=1, linewidth=0.7, zorder=3, facecolor=circle_color,
                     edgecolor="k")
        ax.scatter(eq_ist_x, eq_ist_y, c='w', marker='^', linewidths=0.7, zorder=4, s=20)
        ax.scatter(eq_instr_x, eq_instr_y, c='k', marker='^', linewidths=0.45, zorder=4, s=25)
        plt.grid(True)
        plt.title(u'%s' % head_title, fontdict={'family': 'verdana'})



        plt.savefig(self.path + head_title + '.png', dpi=400)
        plt.close()

    def feats_diagram(self, F, title):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.bar(range(len(F)), F)

        ax.set_ylabel(title)
        ax.set_xlabel('feat')
        ax.set_xticks(range(len(F)), )
        ax.set_xticklabels(self.imp.data_header, rotation='vertical')

        plt.grid(True)
        fig.tight_layout()
        plt.savefig(self.path + title + '.png', dpi=400)
        plt.close()

    def grid_res(self, X, title, r):
        """отображение результатов алгоритма в виде сетки грида"""
        plt.clf()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        ax = plt.gca()
        m = Basemap(llcrnrlat=self.m_c[2], urcrnrlat=self.m_c[3],
                    llcrnrlon=self.m_c[0], urcrnrlon=self.m_c[1],
                    resolution='l')
        m.drawcountries(zorder=1, linewidth=0.6)
        m.drawcoastlines(zorder=1, linewidth=0.6)
        delta = 2
        parallels = np.arange(0., 90, delta)
        m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
        meridians = np.arange(0., 360, delta)
        m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)

        ax.scatter(self.X[:, 0], self.X[:, 1], marker='.', color='k', lw=0, s=5, zorder=0, alpha=.55)

        for xy in X:
            ax.add_artist(
                RegularPolygon(xy=(xy[0], xy[1]), numVertices=4, radius=np.sqrt(r)/10, orientation=math.pi / 4, lw=0,
                               facecolor='b', edgecolor='none', zorder=2, alpha=0.75))

        ax.scatter(self.eq_ist[:, 0], self.eq_ist[:, 1], marker='^', color='r', lw=0.5, zorder=3, s=8)
        ax.scatter(self.eq_instr[:, 0], self.eq_instr[:, 1], marker='o', color='r', lw=0.5, zorder=4, s=8)

        scB = plt.scatter([], [], c='b', linewidth='0.5', label='barrier result', zorder=2)
        scEQis = plt.scatter([], [], c='r', marker='^', linewidth='0.5', label=self.eqLegend[1], zorder=2)
        scEQitr = plt.scatter([], [], c='r', linewidth='0.5', label=self.eqLegend[2], zorder=2)
        plt.legend(handles=[scB, scEQis, scEQitr], loc=8, bbox_to_anchor=(0.5, -0.3), ncol=2)

        plt.title(title)
        plt.savefig(self.path + title + '.png', dpi=500)
        plt.close()

    def node_ln_diff_res(self, SETS, EXT=None, labels=None, title=None, title2=None):
        """отображение разности в алгоритмах"""

        sqrt_pol = self.pol

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        m = Basemap(llcrnrlat=self.m_c[2], urcrnrlat=self.m_c[3],
                    llcrnrlon=self.m_c[0], urcrnrlon=self.m_c[1],
                    resolution='h')
        # m.fillcontinents(color='white', lake_color='aqua',zorder=0, alpha=.5)
        m.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True, zorder=0)

        m.drawcountries(zorder=1, linewidth=0.6)
        m.drawcoastlines(zorder=1, linewidth=0.6)
        parallels = np.arange(0., 90, 2)
        m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
        meridians = np.arange(0., 360, 4)
        m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)

        ax.add_patch(patches.Polygon(sqrt_pol, edgecolor="b", facecolor='none', alpha=0.4, zorder=0))

        if EXT is not None:
            ax.scatter(EXT[:, 0], EXT[:, 1], c='#fd41cd', marker='s', s=20, linewidths=0.0, label='e2xt')

        plt.scatter(self.X[:, 0], self.X[:, 1], marker='.', s=11, c='k', linewidth=0, zorder=1)

        colors = ['b', 'g', 'y']
        legends = []
        for i, SET in enumerate(SETS):
            for x, y, r in zip(SET[:, 0], SET[:, 1], [self.r for i in range(len(SET))]):
                #circle = ax.add_artist(Circle(xy=(x, y),
                #                              radius=r, alpha=0.8, linewidth=0.75, zorder=4, facecolor=colors[i],
                 #                             edgecolor="k", label=labels[i]))

                m.tissot(x, y, r, 50, alpha=0.9, linewidth=0.75, zorder=2, facecolor=colors[i], edgecolor="k")

            legends.append(plt.scatter([], [], c=colors[i], linewidth='0.5', label=labels[i], zorder=2))

        eq_i = plt.scatter(self.eq_ist[:, 0], self.eq_ist[:, 1], c='r', marker='^', linewidths=0.45, zorder=5, s=20,
                           label=self.eqLegend[1], edgecolors='k')
        eq_in = plt.scatter(self.eq_instr[:, 0], self.eq_instr[:, 1], c='r', marker='o', linewidths=0.45, zorder=5,
                            s=20, label=self.eqLegend[1], edgecolors='k')
        legends.append(eq_i)
        legends.append(eq_in)
        plt.legend(handles=legends, loc=8, bbox_to_anchor=(0.5, -0.4), ncol=3)

        plt.title(title, y=1.07)
        plt.suptitle(title2, y=0.83)
        plt.savefig(self.path + title + '.png', dpi=400)
        plt.close()



    def ln_to_grid(self, result, title, r=0.2252):
        """конвертирование окружностей пересечений линеаментов в виде сетки грида"""
        GRID = read_csv(
            path='/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/gridVers/d0.1cut/kvz_coord.csv',
            col=['x', 'y']).T

        def get_grid_around_ln(ln, r):
            cutlnGrid = []
            for i in GRID:
                evk_array = np.zeros((1, len(ln)))
                for n, d in enumerate(i):
                    evk_array += (d - ln[:, n]) ** 2
                evk_gpXY = np.sqrt(evk_array[0])
                evk_wh = np.where(evk_gpXY <= r)[0]
                if len(evk_wh) > 0:
                    cutlnGrid.append(i)
            return np.array(cutlnGrid)

        X = get_grid_around_ln(self.imp.data_coord[result], r=0.15)
        pers = check_pix_pers(X, grid=True)
        acc = acc_check(X, self.imp.eq_all, self.r, grid=True)
        title = 'grid_%s B=%s(%s%s) acc=%s' % (title, len(X), pers, '%', acc)
        print(title)

        plt.clf()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        ax = plt.gca()
        m = Basemap(llcrnrlat=self.m_c[2], urcrnrlat=self.m_c[3],
                    llcrnrlon=self.m_c[0], urcrnrlon=self.m_c[1],
                    resolution='l')
        m.drawcountries(zorder=1, linewidth=0.6)
        m.drawcoastlines(zorder=1, linewidth=0.6)
        parallels = np.arange(0., 90, 2)
        m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
        meridians = np.arange(0., 360, 2)
        m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)

        ax.scatter(GRID[:, 0], GRID[:, 1], marker='.', color='k', lw=0, s=8, zorder=0)
        # ax.scatter(V[:, 0], V[:, 1], marker='s', color='g', lw=0, s=70)

        for xy in X:
            ax.add_artist(
                RegularPolygon(xy=(xy[0], xy[1]), numVertices=4, radius=r - 0.05, orientation=math.pi / 4, lw=0,
                               color='b', zorder=2, alpha=0.75))

        ax.scatter(self.eq_ist[:, 0], self.eq_ist[:, 1], marker='^', color='r', lw=0.5, zorder=3)
        ax.scatter(self.eq_instr[:, 0], self.eq_instr[:, 1], marker='o', color='r', lw=0.5, zorder=4)

        plt.title(title)
        plt.savefig(self.path + title + '.png', dpi=400)
        plt.close()


############################



def check_pix_pers(A, grid=False):
    """расчет площади результата алгоритма в полигоне"""
    fig = plt.figure()
    ax = plt.gca(aspect='equal')
    pol = ParamGlobal().get_squar_poly_coords()

    plt.axis('off')
    plt.xlim(np.min(pol[:, 0]), np.max(pol[:, 0]))
    plt.ylim(np.min(pol[:, 1]), np.max(pol[:, 1]))

    ax.add_patch(patches.Polygon(pol, color='#008000', zorder=1))
    for x, y, r in zip(A[:, 0], A[:, 1], [0.225 for i in range(len(A))]):
        if grid:
            ax.add_artist(RegularPolygon(xy=(x, y), numVertices=4, radius=np.sqrt(2)/10, orientation=math.pi / 4, lw=0,
                                         facecolor='#ff0000', edgecolor='none', zorder=2))
        else:
            ax.add_artist(
                Circle(xy=(x, y), radius=r, alpha=1, linewidth=0, zorder=2, facecolor='#ff0000', edgecolor='#ff0000'))

    fig.canvas.draw()

    #plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/grid_pers.png', dpi=300)
    reso = fig.canvas.get_width_height()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plt.close()
    data = data.reshape((reso[0] * reso[1], 3))

    idx_green_array1 = np.where(data[:, 1] == 128)[0]
    green_array = data[np.where(data[idx_green_array1, 2] == 0)]

    idx_red_array1 = np.where(data[:, 0] == 255)[0]
    red_array = data[np.where(data[idx_red_array1, 1] == 0)]

    r = len(red_array)
    f = len(green_array) + r
    return round(r * 100 / f, 3)


################################################

def visuaMSdiffPix_bw(Aln, Bln, r, direc, title):
    """пиксельная разность 2х алгоритмов"""
    fc = ParamGlobal().get_field_coords()

    def calc_len_pix(data, r):
        lon_0 = np.mean(fc[:2])
        lat_0 = np.mean(fc[2:])
        m = Basemap(llcrnrlat=fc[2], urcrnrlat=fc[3],
                    llcrnrlon=fc[0], urcrnrlon=fc[1],
                    resolution='l', projection='stere', lon_0=lon_0, lat_0=lat_0)

        ax = plt.gca()

        X_x, X_y = m(data[:, 0], data[:, 1])
        h_ln = ax.scatter(X_x, X_y, marker='.', alpha=0, zorder=2)
        b_ln = ax.scatter([], [], marker='o', s=25, alpha=1, c='#969696', linewidth=0.4, zorder=1)
        for x, y, r in zip(data[:, 0], data[:, 1], [r for i in range(len(data))]):
            m.tissot(x, y, r, 50, ax=ax, alpha=1, linewidth=0, zorder=3, facecolor='#ff0000',
                     edgecolor="none")

        plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/diff_tmp.png', bbox_inches='tight', pad_inches=0,
                    dpi=250)
        img = Image.open('/Users/Ivan/Documents/workspace/result/tmp/diff_tmp.png')
        rgb_im = img.convert('RGB')
        data = np.array(rgb_im)
        H, W = len(data), len(data[0])
        return data, W, H

    a_data, W, H = calc_len_pix(Aln, r)
    plt.clf()
    b_data, _, _ = calc_len_pix(Bln, r)
    plt.clf()
    lon_0 = np.mean(fc[:2])
    lat_0 = np.mean(fc[2:])
    m = Basemap(llcrnrlat=fc[2], urcrnrlat=fc[3],
                llcrnrlon=fc[0], urcrnrlon=fc[1],
                resolution='l', projection='stere', lon_0=lon_0, lat_0=lat_0)
    m.drawcountries(zorder=1, linewidth=0.9)
    m.drawcoastlines(zorder=1, linewidth=0.9)
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[0, 0, 0, 0], zorder=1, linewidth=0.4, alpha=0.7)
    meridians = np.arange(0., 360, 2)
    m.drawmeridians(meridians, labels=[0, 0, 0, 0], zorder=1, linewidth=0.4, alpha=0.7)
    ax = plt.gca()

    X_x, X_y = m(Aln[:, 0], Aln[:, 1])
    h_ln = ax.scatter(X_x, X_y, marker='.', alpha=0, zorder=2)
    b_ln = ax.scatter([], [], marker='o', s=25, alpha=1, c='#969696', linewidth=0.4, zorder=1)
    for x, y, r in zip(Aln[:, 0], Aln[:, 1], [r for i in range(len(Aln))]):
        m.tissot(x, y, r, 50, ax=ax, alpha=0, linewidth=0, zorder=3, facecolor='#ff0000',
                 edgecolor="none")

    plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/basemap.png', bbox_inches='tight', pad_inches=0, dpi=250)
    img = Image.open('/Users/Ivan/Documents/workspace/result/tmp/basemap.png')
    rgb_im = img.convert('RGB')
    data_final = np.array(rgb_im)
    pers = 0
    for h in range(H):
        for w in range(W):
            if np.array_equal(a_data[h][w], b_data[h][w]):
                pass
            else:
                pers += 1
                data_final[h][w] = [125, 125, 125]

    print('разность %s%s' % (pers * 100 / (W * H), '%'))

    imsave(direc + title + '.png', data_final)
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], zorder=1, linewidth=0.4, alpha=0.7)
    meridians = np.arange(0., 360, 2)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], zorder=1, linewidth=0.4, alpha=0.7)
    plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/basemap_axis.png', dpi=300)

    plt.close()

def visuaMSdiffPix_color(Aln, Bln, r, direc, title):
    """пиксельная разность 2х алгоритмов"""
    fc = ParamGlobal().get_field_coords()

    original_umask = os.umask(0)
    if not os.path.exists(direc+'/tmp/'):
        os.makedirs(direc+'/tmp/', exist_ok=True)
    os.umask(original_umask)

    def calc_len_pix(data, r):
        lon_0 = np.mean(fc[:2])
        lat_0 = np.mean(fc[2:])
        m = Basemap(llcrnrlat=fc[2], urcrnrlat=fc[3],
                    llcrnrlon=fc[0], urcrnrlon=fc[1],
                    resolution='l')

        ax = plt.gca()

        X_x, X_y = m(data[:, 0], data[:, 1])
        h_ln = ax.scatter(X_x, X_y, marker='.', alpha=0, zorder=2)
        b_ln = ax.scatter([], [], marker='o', s=25, alpha=1, c='#969696', linewidth=0.4, zorder=1)
        for x, y, r in zip(data[:, 0], data[:, 1], [r for i in range(len(data))]):
            m.tissot(x, y, r, 50, ax=ax, alpha=1, linewidth=0, zorder=3, facecolor='#ff0000',
                     edgecolor="none")

        plt.savefig(direc+'/tmp/'+'diff_tmp.png', bbox_inches='tight', pad_inches=0,
                    dpi=250)
        img = Image.open(direc+'/tmp/'+'diff_tmp.png')
        rgb_im = img.convert('RGB')
        data = np.array(rgb_im)
        H, W = len(data), len(data[0])
        return data, W, H

    a_data, W, H = calc_len_pix(Aln, r)
    plt.clf()
    b_data, _, _ = calc_len_pix(Bln, r)
    plt.clf()
    lon_0 = np.mean(fc[:2])
    lat_0 = np.mean(fc[2:])
    m = Basemap(llcrnrlat=fc[2], urcrnrlat=fc[3],
                llcrnrlon=fc[0], urcrnrlon=fc[1],
                resolution='h')
    # m.fillcontinents(color='white', lake_color='aqua',zorder=0, alpha=.5)
    m.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True, zorder=0)
    m.drawcountries(zorder=1, linewidth=0.9)
    m.drawcoastlines(zorder=1, linewidth=0.9)
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[0, 0, 0, 0], zorder=1, linewidth=0.4, alpha=0.7)
    meridians = np.arange(0., 360, 2)
    m.drawmeridians(meridians, labels=[0, 0, 0, 0], zorder=1, linewidth=0.4, alpha=0.7)
    ax = plt.gca()

    X_x, X_y = m(Aln[:, 0], Aln[:, 1])
    h_ln = ax.scatter(X_x, X_y, marker='.', alpha=0, zorder=2)
    b_ln = ax.scatter([], [], marker='o', s=25, alpha=1, c='#969696', linewidth=0.4, zorder=1)
    for x, y, r in zip(Aln[:, 0], Aln[:, 1], [r for i in range(len(Aln))]):
        m.tissot(x, y, r, 50, ax=ax, alpha=0, linewidth=0, zorder=3, facecolor='#ff0000',
                 edgecolor="none")

    plt.savefig(direc+'/tmp/'+'basemap.png', bbox_inches='tight', pad_inches=0, dpi=250)
    img = Image.open(direc+'/tmp/'+'basemap.png')
    rgb_im = img.convert('RGB')
    data_final = np.array(rgb_im)
    pers = 0
    for h in range(H):
        for w in range(W):
            if np.array_equal(a_data[h][w], b_data[h][w]):
                pass
            else:
                pers += 1
                data_final[h][w] = [232, 0, 100]

    print('разность %s%s' % (pers * 100 / (W * H), '%'))

    imsave(direc + title + '.png', data_final)
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], zorder=1, linewidth=0.4, alpha=0.7)
    meridians = np.arange(0., 360, 2)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], zorder=1, linewidth=0.4, alpha=0.7)
    plt.savefig(direc+'/tmp/'+'basemap_axis.png', dpi=300)

    plt.close()


def calc_acc_pixpoly(B, eq_data, delta):
    """расчет точности алгоритма по пикселям"""
    fig = plt.figure()

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax = plt.gca()

    plt.axis('off')

    coord_field = ParamGlobal().get_field_coords()
    plt.xlim(coord_field[0], coord_field[1])
    plt.ylim(coord_field[2], coord_field[3])
    center_x, center_y = np.mean(coord_field[:2]), np.mean(coord_field[2:])

    def calc_len_pix():
        fig.canvas.draw()
        reso = fig.canvas.get_width_height()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape((reso[0] * reso[1], 3))

        idx_red_array1 = np.where(data[:, 0] == 255)[0]
        red_array = data[np.where(data[idx_red_array1, 1] == 0)]

        return len(red_array)

    test_dot = ax.scatter(center_x, center_y, marker='o', c='#ff0000', lw=0, s=70, zorder=2)
    one_dot_leng = calc_len_pix()
    test_dot.set_visible(False)

    for x, y, r in zip(B[:, 0], B[:, 1], [delta for i in range(len(B))]):
        ax.add_artist(RegularPolygon(xy=(x, y), numVertices=4, radius=0.14, orientation=math.pi / 4, lw=0,
                                     facecolor='#ff0000', edgecolor='#ff0000', zorder=1))

    # plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/' + 'field.png', dpi=100)
    zero_r = calc_len_pix()
    w = 0
    acc_points = 0
    miss_points = 0

    for i, xy in enumerate(eq_data):
        r = ax.scatter(xy[0], xy[1], marker='o', c='#ff0000', lw=0, s=70, zorder=2)
        a_r = calc_len_pix()

        if np.abs(zero_r - a_r) < one_dot_leng:
            acc_points += 1
            w += 1
        else:
            miss_points += 1
        r.set_visible(False)

    plt.close()
    return w / len(eq_data), acc_points, miss_points

def read_csv(path, col):
    """чтение csv файла по col колонкам"""
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")
    for i, title in enumerate(col):
        cell = frame[title].values

        try:
            cell = cell[~np.isnan(cell)]
        except Exception as ex:
            print(ex)
            for j, c in enumerate(cell):
                try:
                    np.float(c.replace(',', '.'))
                except:
                    print('Error in row:%s "%s"' % (j, c))

        array.append(cell)

    #return np.array(array).astype(float)
    return np.array(array)

def sample_color_check(node, EQs, range):
    evk = np.zeros((1, len(EQs)))
    for n, d in enumerate(node):
        evk += (d - EQs[:, n]) ** 2
    evk = np.sqrt(evk[0])
    near_sample_range = np.where(evk<=range)[0]
    if len(near_sample_range) > 0:
        return True
    else:
        return False

def acc_check(result, EQ, r, grid=False):

    """вычисление точности алгоритма"""
    accEQ = 0
    if grid:
        r1, r2 = 15.5, 22
    else:
        r1, r2 = r, r*2

    for eq in EQ:
        if len(result) == 0:
            acc = 0
        else:
            evk = np.zeros((1, len(result)))
            for n, d in enumerate(eq):
                evk += (d - result[:, n]) ** 2
            evk = np.sqrt(evk[0])
            b_evk = evk[np.argmin(evk)]
            if b_evk <= r1:
                acc = 1
            elif r1 < b_evk <= r2:
                acc = (r2 - b_evk) / r1
            else:
                acc = 0
        accEQ += acc
    return round(accEQ / len(EQ), 4)
