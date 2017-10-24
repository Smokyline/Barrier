from bmain.alghTools.supportAlg.import_data import ImportData
from bmain.alghTools.supportAlg.drawMap import Visual
from bmain.app.global_param import ParamGlobal
from bmain.barrier.barrier_version import BarrierMod

import os
import numpy as np

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='altai_mk1')


bar = BarrierMod(imp, gp)
r = bar.oneVoneP()

print(r.title)

original_umask = os.umask(0)
vis = Visual(X=imp.data_coord, r=0.225, imp=imp, gp=gp, path=imp.save_path)
if gp.gridVers:
    vis.grid_res(imp.data_coord[r.result], title=r.title, r=2)
else:
    pass
    #vis.visual_circle(res=r.result, title=r.title)





#vis.ln_to_grid(r.result, 'barrier')

#cora_res = imp.read_cora_res(c=1)
cora_res = imp.read_cora_res_2()
vis.visual_circle(res=cora_res, title='cora')

#vis.ln_to_grid(cora_res, 'cora')
#c = CompareAlgh(imp=imp, vis=vis, barrierX=r.result, coraX=cora_res)
#c.visual_compare(result=r)



os.umask(original_umask)
