
'''
    def simple(self):
        X = self.X[:, self.feats]
        Y = self.Y[:, self.feats]
        V = self.V[:, self.feats]
        idxB = Core(X, Y, V, self.param, self.feats).idxB

        return Result(idxB, self.param, self.imp, 'simple')

    def iter_learning(self):
        X = self.X[:, self.feats]
        Y = self.Y[:, self.feats]
        V = self.V[:, self.feats]
        old_idx = None
        while True:
            idxB = Core(X, Y, V, self.param, self.feats).idxB
            V = X[idxB]
            if len(idxB) >= 130:
                break
            old_idx = idxB
        return Result(old_idx, self.param, self.imp, 'iterLearning')

    def adaXoneV(self):
        idxX = np.arange(len(self.X))
        idxV = find_VinXV(self.indexX, idxX, self.indexV)
        itr = 1
        while True:
            print('\niter', itr)
            print('|V|=%s |X|=%s' % (len(idxV), len(idxX)))
            idxNewX = np.array([])
            idxNewV = np.array([])
            idxXwV = np.array([int(i) for i, x in enumerate(idxX) if x not in idxV])

            for v in self.X[idxV]:
                fullXV = np.array([])
                for f in self.feats:
                    feat = [f, ]
                    XF = self.X[idxX][:, feat]
                    VF = np.array([v])[:, feat]
                    idxB = Core(XF, VF, self.param, feat).idxB
                    fullXV = np.append(fullXV, idxB)

                countX = np.array([len(np.where(fullXV == i)[0]) for i in range(len(idxX))])

                UPDcountX = countX[idxXwV]
                UPDidxX = idxX[idxXwV]
                UPDV = np.array([int(idx) for i, idx in enumerate(UPDidxX) if UPDcountX[i] >= max(UPDcountX)]).astype(int)

                idxNewX = np.append(idxNewX, UPDidxX).astype(int)
                idxNewV = np.append(idxNewV, UPDV)

            idxNewV = np.unique(idxNewV).astype(int)
            # idxX = idxXwV[[int(i) for i, x in enumerate(idxNewX) if x not in idxNewV]]
            idxVit = np.union1d(idxV, idxNewV)
            if len(idxVit) > 150:
                return Result(idxV, self.param, self.imp, 'adaXoneV')
            idxV = idxVit
            itr += 1

    def adaXallV(self):
        idxX = np.arange(len(self.X))
        idxV = find_VinXV(self.indexX, idxX, self.indexV)
        itr = 1
        while True:
            print('\niter', itr)
            print('|V|=%s |X|=%s' % (len(idxV), len(idxX)))

            fullXV = np.array([])

            for f in self.feats:
                feat = [f, ]
                XF = self.X[idxX][:, feat]
                VF = self.X[idxV][:, feat]
                idxB = Core(XF, VF, self.param, feat).idxB
                fullXV = np.append(fullXV, idxB)

            countX = np.array([len(np.where(fullXV == i)[0]) for i in range(len(idxX))]).astype(int)
            idxXwV = np.array([int(i) for i, x in enumerate(idxX) if x not in idxV])

            UPDcountX = countX[idxXwV]
            idxNewX = idxX[idxXwV]
            idxNewV = np.array([int(idx) for i, idx in enumerate(idxNewX) if UPDcountX[i] >= max(UPDcountX)]).astype(int)

            idxVit = np.union1d(idxV, idxNewV)

            # idxX = idxNewX[[int(i) for i, x in enumerate(idxNewX) if x not in idxVit]]
            if len(idxVit) > 135:
                return Result(idxV, self.param, self.imp, 'adaXallV')
            idxV = idxVit
            itr += 1

    def allVoneF(self):
        IDX = np.array([])
        for f in self.feats:
            feat = [f, ]
            X = self.X[:, feat]
            V = self.V[:, feat]
            Y = self.Y[:, feat]
            idxB = Core(X, Y, V, self.param, feat).idxB
            IDX = np.append(IDX, idxB)
        countX = np.array([len(np.where(IDX == i)[0]) for i in range(len(self.X))])
        idxB, _ = self.count_border_blade(self.param.border, countX)
        return Result(idxB, self.param, self.imp, 'allVoneF')

    def oneVallF(self):
        IDX = np.array([])
        for v in self.V:
            V = np.array([v])[:, self.feats]
            X = self.X[:, self.feats]
            Y = self.Y[:, self.feats]
            idxB = Core(X, Y, V, self.param, self.feats).idxB
            IDX = np.append(IDX, idxB)
        countX = np.array([len(np.where(IDX == i)[0]) for i in range(len(self.X))])
        idxB, _ = self.count_border_blade(self.param.border, countX)
        return Result(idxB, self.param, self.imp, 'oneVallF')





    def allVoneP_iter(self):
        idxX = np.arange(len(self.X))
        idxV = find_VinXV(self.indexX, idxX, self.indexV)
        itr = 1
        while True:
            fullIDX = np.array([])
            for f in self.feats:
                feat = [f, ]
                XF = self.X[idxX][:, feat]
                VF = self.X[idxV][:, feat]
                YF = self.Y[idxV][:, feat]
                idxB = Core(XF, YF, VF, self.param, feat).idxB
                fullIDX = np.append(fullIDX, idxB)

            countX = np.array([len(np.where(fullIDX == i)[0]) for i in range(len(idxX))]).astype(int)
            idxNewV = self.count_border_blade(self.param.border, countX)
            UPDidxV = np.union1d(idxV, idxNewV)
            if len(UPDidxV) > 140:
                return Result(idxV, self.param, self.imp, 'allVonePiter')
            idxV = UPDidxV
            itr += 1

'''