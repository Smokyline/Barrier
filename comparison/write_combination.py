

import codecs
import os

import numpy as np
from barrier.core import Barrier

from barrier.parameters import ParamGlobal
from module.import_data import ImportData
from module.result import Result


original_umask = os.umask(0)



gp = ParamGlobal()
imp = ImportData(zone=gp.zone, param=gp, gridVers=gp.gridVers, folder_name='kvz_mk2')
imp.set_save_path()
s_array = np.arange(-0.1, -9.1, -0.05)
border_array = np.arange(1, 25, 0.5)
X = imp.data
Y = imp.train

for s_var in s_array:
    for border_var in border_array:
        gp.s = s_var
        gp.border = ['ro', border_var]


        barrier = Barrier(X, Y, gp)

        r = Result(barrier, gp, imp)

        print(r.title)

        acc = r.acc
        hs_count = r.lenB


        ###
        f = codecs.open(imp.save_path + 'result_combination.txt', "a", "utf-8")
        f.write(u'%s %s %s %s\n' % (gp.s, gp.border[1], acc, hs_count))
        f.close()
        ###


os.umask(original_umask)
