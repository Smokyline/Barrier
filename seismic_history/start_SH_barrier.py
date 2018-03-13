import os

from barrier_modules.import_data import ImportData

from barrier_main.barrier_version import BarrierMod
from barrier_main.set_global_param import ParamGlobal
from barrier_modules.drawMap import Visual
from comparison.comparison_two_res import CompareAlgh
from barrier_modules.tools import read_csv_pandas

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='altai_mk1')
EXT = read_csv_pandas('/home/ivan/Documents/workspace/resources/csv/Barrier/altai/altaySayBaikal_EXT.csv')

barier = BarrierMod(imp, gp)
barrier_result = barier.oneVoneP()