import os
from module.result import Result
from module.import_data import ImportData

from barrier.core import Barrier
from barrier.parameters import ParamGlobal

from module.tools import read_csv_pandas

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, param=gp, gridVers=gp.gridVers, folder_name='kvz_mk1')
#imp.EXT = read_csv_pandas('/home/ivan/Documents/workspace/resources/csv/Barrier/altai/altaySayBaikal_EXT.csv')

original_umask = os.umask(0)

#X = imp.data_full[:, [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]]
#Y = imp.data_sample[:, [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]]
X = imp.data
Y = imp.train
barrier = Barrier(X, Y, gp)

barrier_result = Result(barrier, gp, imp)
print(barrier_result.title)

# сохранение результата в csv
#barrier_result.save_res_to_csv()

# карта с высокосейсмичными линеаментами Барьер
barrier_result.visual_barrier()

# карта с высокосейсмичными объектами Коры
#barrier_result.visual_cora(vers=1)

# сравнение Коры и Барьер
#barrier_result.compare_algh()

barrier_result.visual_hs_feats()

os.umask(original_umask)
