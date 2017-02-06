import os
from EPAn.core import *
from EPAn.coreMB import coreS
from alghTools.drawMap import *
from alghTools.tools import read_csv, km

res_dir = '/Users/Ivan/Documents/workspace/resourses/csv/newEPA/'
# res_dir = 'C:\\Users\\smoky\\Documents\workspace\\resourses\\csv\\newEPA\\'

data_full = read_csv(res_dir + 'Caucasus_khar.csv').T
data_sample = read_csv(res_dir + 'Caucasus_sample.csv').T
data_coord = read_csv(res_dir + 'Caucasus_coord.csv', ['x', 'y']).T

idxCX = read_csv(res_dir + 'Caucasus_khar.csv', ['idx'])[0]
idxCV = read_csv(res_dir + 'Caucasus_sample.csv', ['idx'])[0]

class Result:
    def __init__(self, B, V, title):
        self.result = B
        self.V = V
        self.title = title
        self.acc = check_pix_pers(data_coord[self.result])

class Barrier:
    def __init__(self, feats, param):
        self.FEATS = feats
        self.X = data_full
        self.V = data_sample
        self.param = param

    def simple(self):

        idxB, idxV = core(self.X[:, self.FEATS], self.V[:, self.FEATS], idxCX, idxCV, self.FEATS, self.param['q'], self.param['p'], self.param['s'], self.param['bar'],
                          self.param['delta'],self.param['kmeans'], self.param['alphaDem'],self.param['alphaMax'],
                          self.param['pers'], self.param['epsilon'])

        title_param = set_title_param(param_global)
        title = '%s P=%s X(V)=%s(%s%s) V=%s %s' % (
            alg_name, len(self.FEATS), len(idxB), round((len(idxB) * 100 / len(self.X)), 1), '%', len(idxV), title_param)
        return Result(idxB, idxV, title)

    def oneVoneP(self, border):
        fullXV = np.array([])
        for vn, v in enumerate(self.V):
            IDX = np.array([])
            IDXv = np.array([])
            for f in self.FEATS:
                feat = [f]
                XF = self.X[:, feat]
                VF = np.array([v])[:, feat]
                idxB, idxV = core(XF, VF, idxCX, idxCV, feat, self.param['q'], self.param['p'], self.param['s'], self.param['bar'],
                          self.param['delta'],self.param['kmeans'], self.param['alphaDem'],self.param['alphaMax'],
                          self.param['pers'], self.param['epsilon'])
                IDX = np.append(IDX, idxB)
                IDXv = np.append(IDXv, idxV)

            countX = []
            for i in range(len(self.X)):
                countIdx = len(np.where(IDX == i)[0])
                countX.append(countIdx)
            countX = np.array(countX).astype(int)
            IDXv = np.unique(IDXv).astype(int)

            if border[0] == 'h':
                # countX = countX[IDXv]
                # print(np.mean(countX[np.where(countX > 0)]))
                hCalc = np.array([])
                for i, x in enumerate(countX):
                    if i in IDXv:
                        hCalc = np.append(hCalc, x)
                vIDX = parseIdxH(len(data_full), countX, H=border[1], r=self.param['r'], hcalc=hCalc)
            if border[0] == 'pers':
                vIDX = np.array(persRunner(countX, pers=border[1], revers=True)).astype(int)
            if border[0] == 'kmeans':
                vIDX = km(countX, border[1], randCZ=False)[-1]
            xv = np.array(vIDX).astype(int)

            fullXV = np.append(fullXV, xv)

        idxXV = np.unique(fullXV).astype(int)
        idxVV = checkVinXV(idxCX, idxXV, idxCV)

        title_param = set_title_param(param_global)
        title = 'P=%s X(V)=%s V=%s %s=%s %s' % (len(self.FEATS), len(idxXV), len(idxVV),
                                                      border[0], border[1], title_param)
        return Result(idxXV, idxVV, title)


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
    q_dir = directory
    print(q_dir)
    print(title)
    if global_visual:
        visuaMSu(data_coord, data_coord[idxB], data_coord[idxV], r=0.25, title=title, dir=q_dir, visual=False)
    return idxB


def sampleRelearningV(feat, param, title_param):
    X = data_full
    V = data_sample

    q_dir = create_folder(directory, title_param)

    for it in range(10):
        Xx = X[:, feat]
        Vv = V[:, feat]
        idxB, idxV = core(Xx, Vv, idxCX, idxCV, feat, param['q'], param['p'], param['s'], param['bar'], param['delta'],
                          param['kmeans'], param['alphaDem'],
                          param['alphaMax'], param['pers'], param['epsilon'])

        title = '%s it%s P=%s X(V)=%s(%s%s) V=%s %s' % (
            alg_name, it + 1, len(feat), len(idxB), round((len(idxB) * 100 / len(X)), 1), '%', len(idxV), title_param)
        print('\n', title, sep='')

        visuaMSu(data_coord, data_coord[idxB], data_coord[idxV], r=0.225, title=title, dir=q_dir, visual=False)

        V = data_full[idxB]


def votingFEATS(FEATS, param, title_param, border):
    IDX = np.array([])
    for f in FEATS:
        feat = [f]
        X = data_full[:, feat]
        V = data_sample[:, feat]
        idxB, idxV = core(X, V, idxCX, idxCV, feat, param['q'], param['p'], param['s'], param['bar'], param['delta'],
                          param['kmeans'], param['alphaDem'],
                          param['alphaMax'], param['pers'], param['epsilon'])
        IDX = np.append(IDX, idxB)

    cIDX = np.array([])
    for i in range(len(data_full)):
        countIdx = len(np.where(IDX == i)[0])
        cIDX = np.append(cIDX, countIdx)

    if border[0] == 'h':
        finalIDX = parseIdxH(len(data_full), cIDX, H=border[1], r=param['r'])
    if border[0] == 'pers':
        finalIDX = np.array(persRunner(cIDX, pers=border[1], revers=True)).astype(int)
    if border[0] == 'kmeans':
        finalIDX = km(cIDX, border[1], randCZ=False)[-1]

    idxVV = checkVinXV(idxCX, finalIDX, idxCV)

    title = '%s P=%s X(V)=%s(%s%s) V=%s %s=%s %s' % (
        alg_name, len(FEATS), len(finalIDX), round((len(finalIDX) * 100 / len(data_full)), 1), '%', len(idxVV),
        border[0], border[1], title_param)
    print(title)
    q_dir = directory
    if global_visual:
        visuaMSu(data_coord, data_coord[finalIDX], data_coord[idxVV], r=0.225, title=title, dir=q_dir, visual=False)
    return finalIDX


def votingV(FEATS, param, title_param, border):
    X = data_full[:, FEATS]
    V = data_sample[:, FEATS]

    IDX = np.array([])
    for v in V:
        vv = np.array([v])
        idxB, idxV = core(X, vv, idxCX, idxCV, FEATS, param['q'], param['p'], param['s'], param['bar'], param['delta'],
                          param['kmeans'], param['alphaDem'],
                          param['alphaMax'], param['pers'], param['epsilon'])
        IDX = np.append(IDX, idxB)

    cIDX = np.array([])
    for i in range(len(X)):
        countIdx = len(np.where(IDX == i)[0])
        cIDX = np.append(cIDX, countIdx)

    if border[0] == 'h':
        finalIDX = parseIdxH(len(data_full), cIDX, H=border[1], r=param['r'])
    if border[0] == 'pers':
        finalIDX = np.array(persRunner(cIDX, pers=border[1], revers=True)).astype(int)
    if border[0] == 'kmeans':
        finalIDX = km(cIDX, border[1], randCZ=False)[-1]

    idxVV = checkVinXV(idxCX, finalIDX, idxCV)

    title = '%s P=%s X(V)=%s(%s%s) V=%s %s=%s %s' % (
        alg_name, len(FEATS), len(finalIDX), round((len(finalIDX) * 100 / len(data_full)), 1), '%', len(idxVV),
        border[0], border[1], title_param)
    print(title)
    q_dir = directory
    if global_visual:
        visuaMSu(data_coord, data_coord[finalIDX], data_coord[idxVV], r=0.225, title=title, dir=q_dir, visual=False)
    return finalIDX


def adaXoneV(FEATS, param, title_param):
    idxCVit = idxCV.copy()

    q_dir = create_folder(directory, title_param)

    idxFX = np.arange(len(data_coord))
    idxFV = checkVinXV(idxCX, idxFX, idxCVit)

    itr = 1
    while True:
        X = data_full[idxFX]
        V = data_full[idxFV]

        print('\niter', itr + 1)
        print('|V|=%s |X|=%s' % (len(idxFV), len(idxFX)))

        idxEv = np.array([])
        idxXwV = np.array([])

        for v in V:
            vAr = np.array([v])
            fullXV = np.array([])
            for f in FEATS:
                feat = [f]
                XF = X[:, feat]
                VF = vAr[:, feat]
                idxB, idxV = core(XF, VF, idxCX, idxCVit, feat, param['q'], param['p'], param['s'], param['bar'],
                                  param['delta'],
                                  param['kmeans'], param['alphaDem'],
                                  param['alphaMax'], param['pers'], param['epsilon'])
                fullXV = np.append(fullXV, idxB)
            Vvc = np.array([])
            for i in range(len(idxFX)):
                countIdx = len(np.where(fullXV == i)[0])
                Vvc = np.append(Vvc, countIdx)
            # idxXV = km(idxXV, 4, randCZ=False)[-1]
            idxwV, psi = find_psi(idxFX, idxFV, Vvc)
            idxXwv, idxNewVx = psiX(psi, Vvc, idxFX, idxwV)

            idxEv = np.append(idxEv, idxNewVx)
            idxXwV = np.append(idxXwV, idxXwv)

        idxXwV = np.unique(idxXwV).astype(int)
        idxEv = np.unique(idxEv).astype(int)

        idxwV = np.array([]).astype(int)
        for i, x in enumerate(idxXwV):
            if x not in idxEv:
                idxwV = np.append(idxwV, i)
        idxXwV = idxXwV[idxwV]
        idxFX = idxXwV

        idxFV = np.append(idxFV, idxEv)

        updIdxXV = idxFV
        idxVVit = checkVinXV(idxCX, updIdxXV, idxCVit)
        idxCVit = np.append(idxCVit, idxCX[idxEv])

        title = '%s it%i P=%s X(V)=%s(%s%s) V=%s %s' % (
            alg_name, itr + 1, len(FEATS), len(updIdxXV), round((len(updIdxXV) * 100 / len(data_full)), 1), '%',
            len(idxVVit), title_param)
        print(title)
        if global_visual:
            visuaMSu(data_coord, data_coord[updIdxXV], data_coord[idxVVit], r=0.225, title=title, dir=q_dir, visual=False)
        if len(updIdxXV) > 130:
            return updIdxXV
        else:
            itr += 1


def adaXfullV(FEATS, param, title_param):
    idxCVit = idxCV.copy()

    q_dir = create_folder(directory, title_param)

    idxFX = np.arange(len(data_coord))
    idxFV = checkVinXV(idxCX, idxFX, idxCVit)
    updIdxXV = idxFV
    itr = 1
    while True:
        X = data_full[idxFX]
        V = data_full[idxFV]

        print('\niter', itr + 1)
        print('|V|=%s |X|=%s' % (len(idxFV), len(idxFX)))
        fullXV = np.array([])
        for f in FEATS:
            feat = [f]
            XF = X[:, feat]
            VF = V[:, feat]
            idxB, idxV = core(XF, VF, idxCX, idxCVit, feat, param['q'], param['p'], param['s'], param['bar'],
                              param['delta'],
                              param['kmeans'], param['alphaDem'],
                              param['alphaMax'], param['pers'], param['epsilon'])
            fullXV = np.append(fullXV, idxB)

        XVc = np.array([])
        for i in range(len(idxFX)):
            countIdx = len(np.where(fullXV == i)[0])
            XVc = np.append(XVc, countIdx)

        idxwV, psi = find_psi(idxFX, idxFV, XVc)
        idxXwV, idxNewVx = psiX(psi, XVc, idxFX, idxwV)
        print('new V', len(idxNewVx))

        updIdxXV = np.union1d(updIdxXV, idxNewVx)

        idxFX = idxXwV

        idxFV = np.append(idxFV, idxNewVx)
        idxCVit = np.append(idxCVit, idxCX[idxNewVx])

        idxVVit = checkVinXV(idxCX, updIdxXV, idxCVit)

        title = '%s it%i P=%s X(V)=%s(%s%s) V=%s psi=%s %s' % (
            alg_name, itr + 1, len(FEATS), len(updIdxXV), round((len(updIdxXV) * 100 / len(data_full)), 1), '%',
            len(idxVVit),
            psi, title_param)
        print(title)
        if global_visual:
            visuaMSu(data_coord, data_coord[updIdxXV], data_coord[idxVVit], r=0.225, title=title, dir=q_dir, visual=False)
        if len(updIdxXV) > 130:
            return updIdxXV
        else:
            itr += 1


def oneVoneP(FEATS, param, title_param, border):
    q_dir = directory

    X = data_full
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

    if global_visual:
        visuaMSu(data_coord, data_coord[idxXV], data_coord[idxVV], r=0.225, title=title, dir=q_dir, visual=False)

    return idxXV


def barRun():
    dir_title = 'feat=%s delta=%s' % (len(feat), delta)
    q_dir = create_folder(directory, dir_title)

    X = data_full
    V = data_sample

    if vtng:
        IDX = np.array([])
        for f in feat:
            feat = [f]
            idxB, idxV = core(X, V, idxCX, idxCV, feat, q, p, s, bar=bar)
            IDX = np.append(IDX, idxB)
        idxB = parseIdxH(len(data_full), IDX, H=h, r=r)
        idxV = checkVinXV(idxCX, idxB, idxCV)
        title = 'q:%s p:%s s:%s vt:%s h:%s B:%i|%i V:%i' % (q, p, s, vtng, h, len(idxB), len(data_full), len(idxV))
        print('\n', title, sep='')
        visuaMSu(data_coord, data_coord[idxB], data_coord[idxV], r=0.225, title=title, dir=q_dir, visual=False)
    elif simpleV:

        for it in range(iteration):
            fullIDX = np.array([])
            for vn, v in enumerate(V):
                Vv = np.array([v])
                IDX = np.array([])
                for f in feat:
                    ft = f
                    idxB, idxV = core(X, Vv, idxCX, idxCV, ft, q, p, s, bar=bar)
                    IDX = np.append(IDX, idxB)

                vIDX = parseIdxH(len(data_full), IDX, h, r)
                fullIDX = np.append(fullIDX, vIDX)
            fullIDX = np.unique(fullIDX).astype(int)
            idxVV = checkVinXV(idxCX, fullIDX, idxCV)
            V = data_full[fullIDX]
            idxX = fullIDX
            idxV = idxVV
            title = 'it:%i P:%i h:%s q:%s p:%s s:%s r:%s sV:%s B:%i|%i V:%i' % (
                it + 1, len(feat), h, q, p, s, r, simpleV, len(idxX), len(data_full), len(idxV))
            print(title)
            visuaMSu(data_coord, data_coord[idxX], data_coord[idxV], r=0.225, title=title, dir=q_dir, visual=False)

    else:
        for i in range(iteration):
            idxB, idxV = core(X, V, idxCX, idxCV, feat, q, p, s, bar=bar, delta=delta)
            title = 'it:%s q:%s p:%s s:%s vt:%s delta:%s h:%s B:%i|%i V:%i' % (
                i + 1, q, p, s, vtng, delta, h, len(idxB), len(data_full), len(idxV))
            print('\n', title, sep='')
            visuaMSu(data_coord, data_coord[idxB], data_coord[idxV], r=0.225, title=title, dir=q_dir, visual=False)
            V = data_full[idxB]


def fullVonePiter():
    # FEATS = range(1, 21)
    # feat = [1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20]
    FEATS = [1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]

    q = -1
    p = 1
    s = -1

    h = None
    r = 7

    delta = False
    kmeans = False
    alphaDem = False
    alphaMax = False
    pers = 30
    epsilon = False

    dir_title = 'P=%s q=%s p=%s s=%s h=%s r=%s delta=%s' % (len(FEATS), q, p, s, h, r, delta)
    q_dir = create_folder(directory, dir_title)

    X = data_full
    V = data_sample

    itr = 1
    while True:
        fullIDX = np.array([])
        for f in FEATS:
            feat = [f]
            XF = X[:, feat]
            VF = V[:, feat]
            idxB, idxV = core(XF, VF, idxCX, idxCV, feat, q, p, s, bar=False, delta=delta, kmeans=kmeans,
                              alphaDem=alphaDem,
                              alphaMax=alphaMax, pers=pers, epsilon=epsilon)
            fullIDX = np.append(fullIDX, idxB)

        countX = []
        for i in range(len(X)):
            countIdx = len(np.where(fullIDX == i)[0])
            countX.append(countIdx)
        countX = np.array(countX).astype(int)

        idxXV = np.array(persRunner(countX, pers=30, revers=True)).astype(int)
        idxVV = checkVinXV(idxCX, idxXV, idxCV)

        title = '%s it:%s P:%i h:%s B:%i|%i q:%s r:%s s:%s delta:%s  V:%i' % (
            alg_name, itr + 1, len(FEATS), h, len(idxXV), len(data_full), q, r, s, delta, len(idxVV))
        print(title)
        visuaMSu(data_coord, data_coord[idxXV], data_coord[idxVV], r=0.225, title=title, dir=q_dir, visual=False)

        idxNewV = np.array(persRunner(countX, pers=10, revers=True)).astype(int)
        V = data_full[idxNewV]
        if len(idxXV) > 140:
            break
        else:
            itr += 1


def compare_algs(res, c=1):
    XVres = res
    DSres = None

    # % объединения A и D
    persAD = check_pix_pers(data_coord[idxUniAD])

    # пересечение разделить на объединение A D
    compA = tanimoto_check(XVres, DSres, idxXintrRes)

    # точность D
    accD = acc_check(data_coord[DSres])
    # точность объединения AD
    accAD = acc_check(data_coord[idxUniAD])

    # idx A/D и D/A
    idxAwD = idx_diff_runnerAwB(XVres, DSres)
    idxDwA = idx_diff_runnerAwB(DSres, XVres)
    # idx AD-(A/D, D/A) объединение минус пересечения
    idxADwAandD = np.union1d(idxAwD, idxDwA)
    # % разности
    print('объединение %s' % check_pix_pers(data_coord[idxXintrRes]))



    titleA = 'comp %s P=%s A(%s%s) vs D(%s)(%s%s) AD=%s(%s%s) ' % (
        alg_name, len(FEATS_GLOBAL), persA, '%', c, persD, '%', len(idxUniAD), persAD, '%')
    compAlgs_str = 'accA=%s accD=%s accAD=%s compA=%s uniAD=%s AwD=%s DwA=%s' % \
                   (acc_A, accD, accAD, compA, len(idxXintrRes), len(idxAwD), len(idxDwA))
    print(titleA, compAlgs_str)

    idxDV = checkVinXV(idxCX, DSres, idxCV)
    visuaMSu(data_coord, data_coord[DSres], data_coord[idxDV], r=0.225,
             title='Diss |X|=%s P=%s (S=%s%s) diss c=%s accD=%s ' % (len(DSres), len(FEATS_GLOBAL), persD, '%', c, accD),
            dir=directory, visual=False)



class CompareAlgh:
    def __init__(self, barrierX, coraX):
        self.algA = barrierX
        self.algB = coraX

        self.union = np.union1d(self.algA, self.algB)
        self.inters = list(set(self.algA) & set(self.algB))
        self.AwB = idx_diff_runnerAwB(self.algA, self.algB)
        self.BwA = idx_diff_runnerAwB(self.algB, self.algA)

        self.persUnion = check_pix_pers(data_coord[self.union])
        self.persA = check_pix_pers(data_coord[self.algA])
        self.persB = check_pix_pers(data_coord[self.algB])

        self.accA = acc_check(data_coord[self.algA])
        self.accB = acc_check(data_coord[self.algB])
        self.accUnion = acc_check(data_coord[self.union])

    def tanimoto(self):
        """мера Танимото (пересечение\объединение)"""
        return round(len(self.inters) / len(self.union), 2)

    def differ(self):
        """разность (объединение минус пересечение)"""
        return np.union1d(self.AwB, self.BwA)




alg_name = 'oneVPtest'
# directory = 'C:\\Users\\smoky\\Documents\\workspace\\result\\Barrier\\best\\'
directory = '/Users/Ivan/Documents/workspace/result/Barrier/' + alg_name + '/'

if not os.path.exists(directory):
    os.makedirs(directory)

FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]  # 11
#FEATS_GLOBAL = [1, 2, 3, 4, 5, 13, 14, 15] #8
# FEATS_GLOBAL = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17, 18] #14
# FEATS_GLOBAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] #18
# FEATS = [1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]
# FEATS_GLOBAL = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]

global_visual = True
param_global = {
    'q': False,
    'p': 1,
    's': -0.7,  # AlphaMax!!
    'r': False,
    'bar': False,
    'delta': False,
    'kmeans': False,
    'alphaDem': False,
    'alphaMax': False,
    'pers': False,
    'epsilon': False
}


#res = simpleRun(FEATS_GLOBAL, param_global, title_param)
#res = votingFEATS(FEATS_GLOBAL, param_global, title_param, border=['h', 9])
#res = votingV(FEATS_GLOBAL, param_global, title_param,  border=['pers', 55])
#res = oneVoneP(FEATS_GLOBAL, param_global, title_param, border=['h', 8])

# sampleRelearningV(FEATS_GLOBAL, param_global, title_param)
# adaXfullV(FEATS_GLOBAL, param_global, title_param)
# adaXoneV(FEATS_GLOBAL, param_global, title_param)
bar = Barrier(FEATS_GLOBAL, param_global)

r = bar.oneVoneP(border=['h', 8])

c = CompareAlgh(barrierX=r.result, coraX=read_cora_res(idxCX, c=1))


compare_title = 'comp %s P=%s Bar(%s%s) vs Cora(%s%s) U=%s(%s%s) '% (
        alg_name, len(FEATS_GLOBAL), c.persA, '%', c.persB, '%', len(c.union), c.persUnion, '%')

compare_title2 = 'accBar=%s accCora=%s accU=%s tanim=%s BnC=%s B/C=%s C/B=%s' % (
    c.accA, c.accB, c.accUnion, c.tanimoto(), len(c.inters), len(c.AwB), len(c.BwA))

vis = Visual(X=data_coord, r=0.225, path=directory)

vis.diff_res(SETS=[data_coord[c.inters], data_coord[c.AwB], data_coord[c.BwA]], labels=['BnC', 'B/C', 'C/B'], title=compare_title, title2=compare_title2)
vis.color_res(B=data_coord[r.result], title=r.title, V=data_coord[r.V])
vis.color_res(B=data_coord[c.algB], title='Cora-3', V=None)

vis.bw_stere_res(B=data_coord[c.algA], head_title='Барьер', circle_color='#aeaeae')
vis.bw_stere_res(B=data_coord[c.algB], head_title='Кора-3', circle_color='none')
vis.bw_stere_res(B=data_coord[c.union], head_title='Объединение', circle_color='#898989')

visuaMSdiffPix_ras(data_coord[c.union], data_coord[c.inters], r=0.225, title='ras_diff', head_title='Разность', direc=directory)
