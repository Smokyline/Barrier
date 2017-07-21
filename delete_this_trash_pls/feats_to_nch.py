import os
import sys
import numpy as np
import pandas as pd
from bmain.alghTools.tools import read_csv


def _read_txt_param(res_dir):
    path = res_dir + 'param_str.txt'
    f = open(path, 'r')
    f_str = f.read()
    for s in ['[', ']', "'", ',']:
        f_str = f_str.replace(s, '')
    f_list = f_str.split()
    return f_list

def save_to_csv(data, cols, title, path):
    original_umask = os.umask(0)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    Bdf = pd.DataFrame(data, columns=cols)
    Bdf.to_csv(path + title + '.csv', index=False, header=True,
              sep=';', decimal=',')

    os.umask(original_umask)


def to_nch(A, B):
    l_row, l_col = A.shape
    nch_array = np.zeros(A.shape)

    blacklist = [0, 4, 5, 6, 7, 8, 10, 11, 12,  ]

    for c in range(l_col):
        if c in blacklist:
            #print(data[:, c], c)
            nch_array[:, c] = A[:, c]
            continue
        else:

            if np.count_nonzero(A[:, c] < 0) != 0:
                norm_data = A[:, c] - np.min(A[:, c])
                norm_b = B[:, c] - np.min(B[:, c])
            else:
                norm_data = A[:, c]
                norm_b = B[:, c]

            norm_A_data = norm_data + sys.float_info.epsilon
            norm_B_data = norm_b + sys.float_info.epsilon
            for r, a in enumerate(norm_A_data):
                mxa = np.maximum(a, norm_B_data)
                sub_ab = a-norm_B_data
                div = sub_ab/mxa
                #div = div[~np.isnan(div)]
                #div = div[~np.isinf(div)]
                nch_array[r, c] = np.sum(div)/len(norm_B_data)
    return nch_array

save_path = 'crimea37HBM'

res_path = os.path.expanduser('~' + os.getenv("USER") + '/Documents/workspace/resources/csv/Barrier/')
x_folder = 'crimea37'
cols = _read_txt_param(res_path+x_folder+'/')
path_x = res_path+x_folder+'/khar.csv'
path_c = res_path+x_folder+'/coord.csv'

path_v_s = res_path+'kvz/sample.csv'
path_x_s = res_path+'kvz/khar.csv'



vs_feats = read_csv(path_v_s, col=cols).T
xs_feats = read_csv(path_x_s, col=cols).T
x_feats = read_csv(path_x, col=cols).T

x_coord = read_csv(path_c, col=['x', 'y']).T


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

nch_x = to_nch(A=x_feats, B=x_feats)
nch_v = to_nch(A=vs_feats, B=xs_feats)

save_to_csv(nch_v, cols, 'sample', res_path+'%s/' % save_path)
save_to_csv(nch_x, cols, 'khar', res_path+'%s/' % save_path)
save_to_csv(x_coord, ['x', 'y'], 'coord', res_path+'%s/' % save_path)


#eq = read_csv(res_path+'crimea/_instr.csv', col=['x', 'y'])

