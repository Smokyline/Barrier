
from bmain.alghTools.supportAlg.import_data import ImportData
from bmain.alghTools.tools import save_res_coord_to_csv, coord_in_sample, read_csv


path = '/home/ivan/Documents/workspace/result/Barrier/kvz_epa_P=11/csv_res/'
coord_barrier = read_csv(path+'coord_barrier.csv',['x', 'y']).T
coord_epa = read_csv(path+'coord_epa.csv',['x', 'y']).T
coord_uni = []
for i, xy in enumerate(coord_barrier):
    if coord_in_sample(xy, coord_epa):
        coord_uni.append(xy)



save_res_coord_to_csv(coord_uni, 'uni', path)

