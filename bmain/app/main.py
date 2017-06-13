from bmain.alghTools.supportAlg.import_data import ImportData

from bmain.alghTools.supportAlg.drawMap import Visual
from bmain.app.global_param import ParamGlobal
from bmain.barrier.barrier_version import BarrierMod

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='')
bar = BarrierMod(imp, gp)

r = bar.oneVoneP()
#r = bar.oneVoneP_Y()
#r = bar.simple()
#r = bar.oneVoneP()
#r = bar.allVoneF()
#r = bar.adaXoneV()
# r = bar.adaXfullV()
#r = bar.oneVoneP()


print(r.title)

vis = Visual(X=imp.data_coord, r=0.225, imp=imp, path=imp.save_path)
if gp.gridVers:
    vis.grid_res(imp.data_coord[r.result], title=r.title, r=2)
else:
    vis.visual_circle(res=r.result, title=r.title)




#vis.ln_to_grid(r.result, 'barrier')

#cora_res = read_cora_res(imp.indexX, c=1)
#vis.ln_to_grid(cora_res, 'cora')
#c = CompareAlgh(imp=imp, vis=vis, barrierX=r.result, coraX=cora_res)
#c.visual_compare(result=r)