import os

from module.import_data import ImportData
from barrier.barrier import Barrier

from barrier.parameters import ParamGlobal
from module.drawMap import Visual
from comparison.comparison_two_res import CompareAlgh
from module.tools import read_csv_pandas

gp = ParamGlobal()
imp = ImportData(zone=gp.zone, ln_field=gp.ln_field, gridVers=gp.gridVers, folder_name='altai_mk1')
EXT = read_csv_pandas('/home/ivan/Documents/workspace/resources/csv/Barrier/altai/altaySayBaikal_EXT.csv')

barier = Barrier(imp, gp)
barrier_result = barier.sample_union()