import os
from EPAn.core import *
from EPAn.coreMB import coreS
from alghTools.drawMap import *
from alghTools.tools import read_csv, km

# res_dir = 'C:\\Users\\smoky\\Documents\workspace\\resourses\\csv\\newEPA\\'

delta = 0.2
gridName = 'kvz_khar'
coordGridName = 'kvz_coord'
sampleGridName = 'kvz_sample'
res_dir = '/Users/Ivan/Documents/workspace/resourses/csv/newEPA/gridVers/d%s/' % delta

tp = ['idx', 'Hmax', 'Hmin', 'DH', 'DH/l', 'Hdisp', 'Bmax', 'Bmin', 'DB', 'Bdisp', 'Mmax', 'Mmin', 'DM', 'Mdisp']
data_full = read_csv(res_dir + gridName+'.csv', tp).T
data_sample = read_csv(res_dir + sampleGridName+'.csv', tp).T
data_coord = read_csv(res_dir + coordGridName+'.csv', ['x', 'y']).T

idxCX = read_csv(res_dir + gridName+'.csv', ['idx'])[0]
idxCV = read_csv(res_dir + sampleGridName+'.csv', ['idx'])[0]


def simpleRun(feat, param, title_param):
    X = data_full[:, feat]
    V = data_sample[:, feat]
    idxB, idxV = core(X, V, idxCX, idxCV, feat, param['q'], param['p'], param['s'], param['bar'], param['delta'],
                      param['kmeans'], param['alphaDem'],
                      param['alphaMax'], param['pers'], param['epsilon'])
    # visualLN(idxB, idxV, data_coord, 'kmeans2ksum', directory)

    title = '%s P=%s X(V)=%s(%s%s) V=%s %s' % (
        alg_name, len(feat), len(idxB), round((len(idxB) * 100 / len(X)), 1), '%', len(idxV), title_param)

    # q_dir = create_folder(directory, title)

    print(title)
    return idxB, idxV, title


def oneVoneP(FEATS, param, title_param, border):
    q_dir = directory

    X = data_full  #!
    V = data_sample

    fullXV = np.array([])
    for vn, v in enumerate(V):
        IDX = np.array([])
        #idxOneCV = idxCX[np.where(idxCX == idxCV[vn])]
        #IDXv = np.where(idxCX == idxOneCV)[0]
        #print(IDXv)
        IDXv = np.array([])
        for f in FEATS:
            feat = [f]
            XF = X[:, feat]
            VF = np.array([v])[:, feat]
            idxB, idxV = core(XF, VF, idxCX, idxCV, feat, param['q'], param['p'], param['s'], param['bar'],
                              param['delta'],
                              param['kmeans'], param['alphaDem'],
                              param['alphaMax'], param['pers'], param['epsilon'])
            IDX = np.append(IDX, idxB)
            IDXv = np.append(IDXv, idxV)

        countX = []
        for i in range(len(X)):
            countIdx = len(np.where(IDX == i)[0])
            countX.append(countIdx)
        countX = np.array(countX).astype(int)
        IDXv = np.unique(IDXv).astype(int)

        if border[0] == 'h':
            #countX = countX[IDXv]
            #print(np.mean(countX[np.where(countX > 0)]))
            hCalc = np.array([])
            for i, x in enumerate(countX):
                if i in IDXv:
                    hCalc = np.append(hCalc, x)
            vIDX = parseIdxH(len(data_full), countX, H=border[1], r=param['r'], hcalc=hCalc)
        if border[0] == 'pers':
            vIDX = np.array(persRunner(countX, pers=border[1], revers=True)).astype(int)
        if border[0] == 'kmeans':
            vIDX = km(countX, border[1], randCZ=False)[-1]
        xv = np.array(vIDX).astype(int)

        fullXV = np.append(fullXV, xv)

    idxXV = np.unique(fullXV).astype(int)
    idxVV = checkVinXV(idxCX, idxXV, idxCV)
    Apix = check_pix_pers(data_coord[idxXV])
    title = 'P=%s X(V)=%s(%s%s) V=%s %s=%s %s' % (len(FEATS), len(idxXV), Apix, '%', len(idxVV),
        border[0], border[1], title_param)
    print(title)


    return idxXV, idxVV, title



alg_name = 'oneVoneP'
directory = '/Users/Ivan/Documents/workspace/result/gridBarrier/' + alg_name + '/'

if not os.path.exists(directory):
    os.makedirs(directory)

FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

global_visual = True
param_global = {
    'q': False,
    'p': 1,
    's': -1.1,  # AlphaMax!!
    'r': False,
    'bar': False,
    'delta': False,
    'kmeans': False,
    'alphaDem': False,
    'alphaMax': False,
    'pers': False,
    'epsilon': False
}
title_param = set_title_param(param_global)

#idxX, idxV, title = simpleRun(FEATS_GLOBAL, param_global, title_param)
idxX, idxV, title = oneVoneP(FEATS_GLOBAL, param_global, title_param, border=['h', 13])

acc_A = acc_check(data_coord[idxX])
print('%s acc:%s' % (alg_name, acc_A))

#visual_grid_res(data_coord[idxX], data_coord[idxV], title+'acc:%s' % acc_A, directory)
visuaMSu(data_coord, data_coord[idxX], data_coord[idxV], r=0.225,
         title=title+'acc:%s' % acc_A,
         dir=directory, visual=False)