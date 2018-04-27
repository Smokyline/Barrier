

import codecs
import os

import numpy as np
from barrier_main.barrier import Barrier

from barrier_main.parameters import ParamGlobal
from barrier_modules.import_data import ImportData

original_umask = os.umask(0)



gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='altai_mk1')
imp.set_save_path()
s_array = np.arange(-0.1, -9.1, -0.05)
border_array = np.arange(1, 25, 0.5)

for s_var in s_array:
    for border_var in border_array:
        gp.s = s_var
        gp.border = ['ro', border_var]

        bar = Barrier(imp, gp)
        r = bar.sample_objects_run()
        print(r.title)

        acc = r.acc
        hs_count = r.lenB


        ###
        f = codecs.open(imp.save_path + 'result_combination.txt', "a", "utf-8")
        f.write(u'%s %s %s %s\n' % (gp.s, gp.border[1], acc, hs_count))
        f.close()
        ###


os.umask(original_umask)
