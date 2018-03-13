import os

from barrier_modules.import_data import ImportData

from barrier_main.barrier_version import BarrierMod
from barrier_main.set_global_param import ParamGlobal
from barrier_modules.drawMap import Visual
from comparison.comparison_two_res import CompareAlgh
from barrier_modules.tools import read_csv_pandas

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='altai_mk6')
EXT = read_csv_pandas('/home/ivan/Documents/workspace/resources/csv/Barrier/altai/altaySayBaikal_EXT.csv')


bar = BarrierMod(imp, gp)
r = bar.oneVoneP()

print(r.title)

original_umask = os.umask(0)
vis = Visual(X=imp.data_coord, imp=imp, gp=gp, path=imp.save_path)
if gp.gridVers:
    vis.grid_res(imp.data_coord[r.result], title=r.title, r=2)
else:
    pass
    vis.visual_circle(res=r.result, EXT=EXT, title=r.title)




#vis.ln_to_grid(r.result, 'barrier_main')

#cora_res = imp.read_cora_res(c=1)
#cora_res = imp.read_cora_res_2()

#vis.visual_circle(res=cora_res, EXT=EXT, title='cora')

#vis.ln_to_grid(cora_res, 'cora')
#c = CompareAlgh(imp=imp, vis=vis, barrierX=r.result, coraX=cora_res)
#c.visual_compare(result=r, EXT=EXT)



os.umask(original_umask)
