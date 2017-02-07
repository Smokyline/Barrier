import os
from testing.findingBest.combinations import alg_stack, imp
import numpy as np


barrier_vars = list(alg_stack.keys())
res_path = imp.save_path
print(barrier_vars)
print(res_path)

alg_res = np.empty((0, 6))

for key in barrier_vars:
    print(key, end='... ')
    try:
        txt_file = open(res_path + key + '.txt')
        for line in txt_file:
            parced_line = line.split(' ')
            parced_line[5] = parced_line[5][:-1]
            if parced_line[4] not in alg_res[:, 4]:
                alg_res = np.append(alg_res, np.array([parced_line]), axis=0)
        print('read')
    except:
        print('no %s in folder %s' % (key, res_path))

idx = np.where(np.logical_and(alg_res[:, 1].astype(int) < 140, alg_res[:, 1].astype(int) > 80))[0]
alg_res1 = alg_res[idx]

acc = alg_res1[:, 2].astype(float)
idx_max = np.argsort(acc)
alg_res1[idx_max[-25:]]