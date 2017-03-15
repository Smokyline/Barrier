import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Circle
from matplotlib.patches import RegularPolygon
import matplotlib.patches as patches
import math

from PIL import Image
from skimage.color import rgb2gray
from testing.alghTools.tools import read_csv
import scipy.misc


def get_field_coords():
    return [36, 52, 37, 46]


class Visual:
    def __init__(self, X, r, imp, path):
        self.X = X
        self.imp = imp
        self.path = path
        self.m_c = get_field_coords()
        self.r = r
        self.eq_all, self.eq_ist, self.eq_instr, self.eqLegend = imp.eq_stack()

    def color_res(self, res, title):
        B = self.imp.data_coord[res]
        #V = self.imp.data_coord[v_res]
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        m = Basemap(llcrnrlat=self.m_c[2], urcrnrlat=self.m_c[3],
                    llcrnrlon=self.m_c[0], urcrnrlon=self.m_c[1],
                    resolution='l')
        m.drawcountries(zorder=1, linewidth=0.6)
        m.drawcoastlines(zorder=1, linewidth=0.6)
        parallels = np.arange(0., 90, 2)
        m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
        meridians = np.arange(0., 360, 2)
        m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)

        plt.scatter(self.X[:, 0], self.X[:, 1], c='k', marker='.', lw=0, zorder=0, s=8)

        for x, y, r in zip(B[:, 0], B[:, 1], [self.r for i in range(len(B))]):
            circle_B = ax.add_artist(Circle(xy=(x, y),
                                           radius=r, alpha=0.8, linewidth=0.75, zorder=2, facecolor='b',
                                           edgecolor="k"))

        """if V is not None:
            for x, y, r in zip(V[:, 0], V[:, 1], [self.r for i in range(len(V))]):
                circle_V = ax.add_artist(Circle(xy=(x, y),
                                                radius=r, alpha=0.9, linewidth=0.75, zorder=3, facecolor="g",
                                                edgecolor="k"))"""

        plt.scatter(self.eq_ist[:, 0], self.eq_ist[:, 1], c='r', marker='^', linewidths=0.45, zorder=4, s=20)
        plt.scatter(self.eq_instr[:, 0], self.eq_instr[:, 1], c='r', marker='o', linewidths=0.45, zorder=4, s=20)

        scB = plt.scatter([], [], c='b', linewidth='0.5', label='X(V)', zorder=2)
        scEQis = plt.scatter([], [], c='r', marker='^', linewidth='0.5', label=self.eqLegend[1], zorder=2)
        scEQitr = plt.scatter([], [], c='r', linewidth='0.5', label=self.eqLegend[2], zorder=2)
        plt.legend(handles=[scB, scEQis, scEQitr], loc=8, bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.title(title)
        plt.savefig(self.path + title + '.png', dpi=400)
        plt.close()

    def bw_stere_res(self, B, head_title, circle_color):
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

    def diff_res(self, SETS, labels, title, title2):
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        m = Basemap(llcrnrlat=self.m_c[2], urcrnrlat=self.m_c[3],
                    llcrnrlon=self.m_c[0], urcrnrlon=self.m_c[1],
                    resolution='l')
        m.drawcountries(zorder=1, linewidth=0.9)
        m.drawcoastlines(zorder=1, linewidth=0.9)
        parallels = np.arange(0., 90, 2)
        m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
        meridians = np.arange(0., 360, 2)
        m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)

        plt.scatter(self.X[:, 0], self.X[:, 1], marker='.', s=11, c='k', linewidth=0, zorder=1)

        colors = ['b', 'g', 'y']
        legends = []
        for i, SET in enumerate(SETS):
            for x, y, r in zip(SET[:, 0], SET[:, 1], [self.r for i in range(len(SET))]):
                circle = ax.add_artist(Circle(xy=(x, y),
                                               radius=r, alpha=0.8, linewidth=0.75, zorder=4, facecolor=colors[i],
                                               edgecolor="k", label=labels[i]))
            legends.append(plt.scatter([], [], c=colors[i], linewidth='0.5', label=labels[i], zorder=2))

        eq_i = plt.scatter(self.eq_ist[:, 0], self.eq_ist[:, 1], c='r', marker='^', linewidths=0.45, zorder=5, s=20, label=self.eqLegend[1])
        eq_in = plt.scatter(self.eq_instr[:, 0], self.eq_instr[:, 1], c='r', marker='o', linewidths=0.45, zorder=5, s=20, label=self.eqLegend[1])
        legends.append(eq_i)
        legends.append(eq_in)
        plt.legend(handles=legends, loc=8, bbox_to_anchor=(0.5, -0.3), ncol=3)

        plt.title(title, y=1.07)
        plt.suptitle(title2, y=0.83)
        plt.savefig(self.path + title + '.png', dpi=400)
        plt.close()

    def grid_res(self, X, title, r):
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

        ax.scatter(self.X[:, 0], self.X[:, 1], marker='.', color='k', lw=0, s=8, zorder=0)
        #ax.scatter(V[:, 0], V[:, 1], marker='s', color='g', lw=0, s=70)

        for xy in X:
            ax.add_artist(RegularPolygon(xy=(xy[0], xy[1]), numVertices=4, radius=r-0.05, orientation=math.pi/4, lw=0,
                                           color='b', zorder=2, alpha=0.75))

        ax.scatter(self.eq_ist[:, 0], self.eq_ist[:, 1], marker='^', color='r', lw=0.5, zorder=3)
        ax.scatter(self.eq_instr[:, 0], self.eq_instr[:, 1], marker='o', color='r', lw=0.5, zorder=4)

        plt.title(title)
        plt.savefig(self.path + title + '.png', dpi=400)
        plt.close()




def visuaMSdiffPix_ras(Aln, Bln, r, direc, title, head_title):

    fc = get_field_coords()

    pol = [[fc[0], fc[2]], [fc[0], fc[3]], [fc[1], fc[3]], [fc[1], fc[2]]]

    def calc_len_pix(data, r):
        lon_0 = np.mean(fc[:2])
        lat_0 = np.mean(fc[2:])
        m = Basemap(llcrnrlat=fc[2], urcrnrlat=fc[3],
                    llcrnrlon=fc[0], urcrnrlon=fc[1],
                    resolution='l', projection='stere', lon_0=lon_0, lat_0=lat_0)


        ax = plt.gca()

        X_x, X_y = m(data[:, 0], data[:, 1])
        h_ln = ax.scatter(X_x, X_y, marker='.', alpha=0,  zorder=2)
        b_ln = ax.scatter([], [], marker='o', s=25, alpha=1, c='#969696', linewidth=0.4, zorder=1)
        for x, y, r in zip(data[:, 0], data[:, 1], [r for i in range(len(data))]):
            m.tissot(x, y, r, 50, ax=ax, alpha=1, linewidth=0, zorder=3, facecolor='#ff0000',
                     edgecolor="none")



        plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/test.png', bbox_inches='tight', pad_inches=0, dpi=250)
        img = Image.open('/Users/Ivan/Documents/workspace/result/tmp/test.png')
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
    #parallels = np.arange(0., 90, 2)
    #m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
    #meridians = np.arange(0., 360, 2)
    #m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)

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
                #final_data[h, w] = [255, 255, 255]
                pass
            else:
                pers += 1
                data_final[h][w] = [125, 125, 125]

    print('разность %s%s' % (pers * 100 / (W * H), '%'))

    #plt.imshow(data_final, extent=[fc[0], fc[1], fc[2], fc[3]], zorder=2)


    #plt.grid(True)
    #plt.title(u'%s' % head_title, fontdict={'family': 'verdana'})

    scipy.misc.imsave(direc + title + '.png', data_final)
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], zorder=1, linewidth=0.4, alpha=0.7)
    meridians = np.arange(0., 360, 2)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], zorder=1, linewidth=0.4, alpha=0.7)
    plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/basemap_axis.png', dpi=300)

    plt.close()

    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/3.png', bbox_inches=extent)
############################




def check_pix_pers(A, grid=False):
    fig = plt.figure()
    ax = plt.gca(aspect='equal')

    fc = get_field_coords()

    plt.axis('off')
    plt.xlim(fc[0], fc[1])
    plt.ylim(fc[2], fc[3])

    pol = [[fc[0], fc[2]], [fc[0], fc[3]], [fc[1], fc[3]], [fc[1], fc[2]]]

    ax.add_patch(patches.Polygon(pol, color='#008000', zorder=1))
    for x, y, r in zip(A[:, 0], A[:, 1], [0.225 for i in range(len(A))]):
        if grid:
            ax.add_artist(RegularPolygon(xy=(x, y), numVertices=4, radius=0.21, orientation=math.pi / 4, lw=0,
                                         facecolor='#ff0000', edgecolor='#ff0000', zorder=2))
        else:
            ax.add_artist(Circle(xy=(x, y), radius=r, alpha=1, linewidth=0, zorder=2, facecolor='#ff0000', edgecolor='#ff0000'))

    fig.canvas.draw()

    reso = fig.canvas.get_width_height()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape((reso[0] * reso[1], 3))

    idx_green_array1 = np.where(data[:, 1] == 128)[0]
    green_array = data[np.where(data[idx_green_array1, 2] == 0)]

    idx_red_array1 = np.where(data[:, 0] == 255)[0]
    red_array = data[np.where(data[idx_red_array1, 1] == 0)]

    r = len(red_array)
    f = len(green_array) + r
    plt.close()
    return round(r * 100 / f, 3)



def calc_acc_pixpoly(B, eq_data,  delta):
    fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax = plt.gca()

    plt.axis('off')

    coord_field = get_field_coords()
    plt.xlim(coord_field[0], coord_field[1])
    plt.ylim(coord_field[2], coord_field[3])
    center_x, center_y = np.mean(coord_field[:2]), np.mean(coord_field[2:])

    def calc_len_pix():
        fig.canvas.draw()
        reso = fig.canvas.get_width_height()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape((reso[0] * reso[1], 3))

        #idx_green_array1 = np.where(data[:, 1] == 128)[0]
        #green_array = data[np.where(data[idx_green_array1, 2] == 0)]

        idx_red_array1 = np.where(data[:, 0] == 255)[0]
        red_array = data[np.where(data[idx_red_array1, 1] == 0)]

        return len(red_array)


    #ax.add_patch(patches.Polygon(pols, color='#008000', zorder=0))

    test_dot = ax.scatter(center_x, center_y, marker='o', c='#ff0000', lw=0, s=70, zorder=2)
    one_dot_leng = calc_len_pix()
    test_dot.set_visible(False)

    #ax.scatter(B[:, 0], B[:, 1], c='#ff0000', marker='s', s=100, linewidths=0.0, alpha=1, zorder=1)
    for x, y, r in zip(B[:, 0], B[:, 1], [delta for i in range(len(B))]):
        ax.add_artist(RegularPolygon(xy=(x, y), numVertices=4, radius=r-0.05, orientation=math.pi / 4, lw=0,
                                 facecolor='#ff0000', edgecolor='#ff0000', zorder=1))

    #plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/' + 'field.png', dpi=100)

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
            #plt.savefig('/Users/Ivan/Documents/workspace/result/tmp/' + 'figD' +str(i+1)+ '.png', dpi=100)
            miss_points += 1
        r.set_visible(False)

    plt.close()
    return w / len(eq_data), acc_points, miss_points