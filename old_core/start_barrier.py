import os
from old_core.barrier import Barrier
from module.result import Result

from module.import_data import ImportData

from old_core.parameters import ParamGlobal

from module.tools import read_csv_pandas

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='kvz_mk1')
#imp.EXT = read_csv_pandas('/home/ivan/Documents/workspace/resources/csv/Barrier/altai/altaySayBaikal_EXT.csv')

original_umask = os.umask(0)


barrier = Barrier(imp, gp)
barrier.sample_union()
barrier_result = Result(barrier, gp, imp)
print(barrier_result.title)

# сохранение результата в csv
#barrier_result.save_res_to_csv()

# карта с высокосейсмичными линеаментами Барьер
#barrier_result.visual_barrier()

# карта с высокосейсмичными объектами КОры
#barrier_result.visual_cora(vers=1)

# сравнение Коры и Барьер
#barrier_result.compare_algh()


os.umask(original_umask)
