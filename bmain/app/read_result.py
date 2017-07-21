from bmain.alghTools.supportAlg.import_data import ImportData
from bmain.alghTools.supportAlg.drawMap import Visual
from bmain.app.global_param import ParamGlobal
from bmain.alghTools.tools import read_csv
from bmain.app.result_prepair import Result
import os
import numpy as np


def find_idx_in_coord(res_coord, all_coord):
    eps = 0.00000006
    idx = []

    for xy in res_coord:
        x_dim = np.where(abs(all_coord[:, 0] - xy[0]) < eps)[0]
        if len(x_dim) > 0:
            y_dim = np.where(abs(all_coord[x_dim, 1] - xy[1]) < eps)[0]
            if len(y_dim) > 0:
                idx.append(x_dim[y_dim])

    return np.unique(idx)

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='')

path_to_result = '/home/ivan/Documents/workspace/result/Barrier/epa_res/'
result_coord = read_csv(path_to_result+'coord_14.csv', col=['x', 'y']).T

idx = find_idx_in_coord(result_coord, imp.data_coord)
print(len(idx))
r = Result(idx, None, None, gp, imp, alg_name='B14')

original_umask = os.umask(0)
vis = Visual(X=imp.data_coord, r=0.225, imp=imp, gp=gp, path=imp.save_path)
if gp.gridVers:
    vis.grid_res(imp.data_coord[r.result], title=r.title, r=2)
else:
    vis.visual_circle(res=r.result, title=r.title)
os.umask(original_umask)



