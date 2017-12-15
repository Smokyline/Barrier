

import os

import numpy as np
from barrier_modules.import_data import ImportData

from barrier_main.set_global_param import ParamGlobal

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='altai_mk1')
imp.set_save_path()

original_umask = os.umask(0)


# s border acc lenB
f_name = 'result_combination.txt'
with open(imp.save_path+f_name) as f:
    lines = f.readlines()

res_comb = np.empty((0, 4)).astype(float)
for line in lines:
    line = line[:-1]
    line = line.split(' ')
    res_comb = np.append(res_comb, [np.array(line).astype(float)], axis=0)


res_comb = res_comb[np.where(res_comb[:, 3] < 40)[0]]
res_comb = res_comb[np.where(res_comb[:, 3] > 16)[0]]
res_comb = res_comb[res_comb[:,2].argsort()]
print(res_comb[-20:])



os.umask(original_umask)
