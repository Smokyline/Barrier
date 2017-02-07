import numpy as np
import pandas as pd
from testing.alghTools.tools import read_csv, find_VinXV


data_coord = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/newEPA/Caucasus_coord.csv', ['x', 'y']).T
gp_data = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/newEPA/kvz_grav.csv', ['x', 'y', 'value']).T

idxCX = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/newEPA/Caucasus_khar.csv', ['idx'])[0]
idxCV = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/newEPA/Caucasus_sample.csv', ['idx'])[0]

def FooMy(xy):
    evk_array = np.zeros((1, len(gp_data)))
    for n, d in enumerate(xy):
        evk_array += (d - gp_data[:, n]) ** 2
    evk_array = np.sqrt(evk_array[0])
    idxY = np.where(evk_array <= r)[0]
    Emy = np.sum(gp_data[idxY, 2]) / len(idxY)
    Dmy = np.sum(np.abs(gp_data[idxY, 2] - Emy)) / len(idxY)
    return Dmy, idxY

def Fooy(idxY):
    Y = gp_data[idxY, :2]
    E = 0
    for i, y in enumerate(Y):
        evk_array = np.zeros((1, len(gp_data)))
        for n, d in enumerate(y):
            evk_array += (d - gp_data[:, n]) ** 2
        evk_array = np.sqrt(evk_array[0])
        idxYy = np.where(evk_array <= r)[0]
        Ey = np.sum(gp_data[idxYy, 2]) / len(idxYy)
        Dy = np.sum(np.abs(gp_data[idxYy, 2] - Ey)) / len(idxYy)
        E += Dy
    return E / len(idxY)

def nch(Dy, Dmy):
    return (Dmy - Dy) / max(Dmy, Dy)


gp_par = np.empty((0, 2))
r = 0.2252
for i, my in enumerate(data_coord):
    print(i+1)
    Dmy, idxY = FooMy(my)
    Dy = Fooy(idxY)
    nchMy = nch(Dy, Dmy)
    gp_par = np.vstack((gp_par, [nchMy, idxCX[i]]))
fullData_PlusGP = np.append(data_coord, gp_par, axis=1)

idxXV = np.arange(len(data_coord))
idxVV = find_VinXV(idxCX, idxXV, idxCV)

sampleData_PlusGP = fullData_PlusGP[idxVV]

direc = '/Users/Ivan/Documents/workspace/resourses/'
Adf = pd.DataFrame(fullData_PlusGP, columns=['x', 'y', 'Bdisp', 'idx'])
Adf.to_csv(direc + 'coord_fKVZ.csv', index=False, header=True,
              sep=';', decimal=',')

Bdf = pd.DataFrame(sampleData_PlusGP, columns=['x', 'y', 'Bdisp', 'idx'])
Bdf.to_csv(direc + 'coord_samKVZ.csv', index=False, header=True,
              sep=';', decimal=',')

