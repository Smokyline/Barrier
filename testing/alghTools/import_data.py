import os
import numpy as np
from testing.alghTools.tools import read_csv


class ImportData:
    def __init__(self, zone='', gridVers=False):
        if os.name == 'nt':
            res_dir = 'C:\\Users\\smoky\\Documents\workspace\\resources\\csv\\Barrier\\'
            eq_dir = 'C:\\Users\\smoky\\Documents\\workspace\\resources\\csv\\geop\\kvz\\'
        elif os.name == 'posix':
            eq_dir = '/Users/Ivan/Documents/workspace/resources/csv/geop/kvz/'
            if gridVers:
                res_dir = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/gridVers/d0.2/'
            else:
                res_dir = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/kvz_upd/'

        else:
            print('os not supported')

        col = ['idx', 'Hmax', 'Hmin', 'Hdelta', 'Hgrad', 'Hdisp', 'Hnch', 'Hcnt', 'Hw3-1', 'Hw3-2'
, 'Hw3-3', 'Hw4-1', 'Hw4-2', 'Hw4-3', 'Hw4-4', 'Hw5-1', 'Hw5-2', 'Hw5-3', 'Hw5-4'
, 'Hw5-5', 'Hw6-1', 'Hw6-2', 'Hw6-3', 'Hw6-4', 'Hw6-5', 'Hw6-6', 'Bmax', 'Bmin'
, 'Bdelta', 'Bgrad', 'Bdisp', 'Bnch', 'Bcnt', 'Bw3-1', 'Bw3-2', 'Bw3-3', 'Bw4-1'
, 'Bw4-2', 'Bw4-3', 'Bw4-4', 'Bw5-1', 'Bw5-2', 'Bw5-3', 'Bw5-4', 'Bw5-5', 'Bw6-1'
, 'Bw6-2', 'Bw6-3', 'Bw6-4', 'Bw6-5', 'Bw6-6', 'Mmax', 'Mmin', 'Mdelta', 'Mgrad'
, 'Mdisp', 'Mnch', 'Mcnt', 'Mw3-1', 'Mw3-2', 'Mw3-3', 'Mw4-1', 'Mw4-2', 'Mw4-3'
, 'Mw4-4', 'Mw5-1', 'Mw5-2', 'Mw5-3', 'Mw5-4', 'Mw5-5', 'Mw6-1', 'Mw6-2', 'Mw6-3'
, 'Mw6-4', 'Mw6-5', 'Mw6-6']

        self.data_full = read_csv(res_dir + 'kvz_khar.csv', col).T
        self.data_field = read_csv(res_dir + 'kvz_field.csv', col).T
        self.data_sample = read_csv(res_dir + 'kvz_sample.csv', col).T
        self.data_coord = read_csv(res_dir + 'kvz_coord.csv', ['x', 'y']).T
        self.indexX = np.array(self.data_full[:, 0]).astype(int)
        self.indexV = np.array(self.data_sample[:, 0]).astype(int)

        file_name_all = 'kvz_eq6_all.csv'
        file_name_ist = 'kvz_eq6_istor.csv'
        file_name_inst = 'kvz_eq6_instr.csv'
        self.eq_all = read_csv(eq_dir + file_name_all, ['x', 'y']).T
        self.eq_ist = read_csv(eq_dir + file_name_ist, ['x', 'y']).T
        self.eq_inst = read_csv(eq_dir + file_name_inst, ['x', 'y']).T

    def eq_stack(self):
        legend = ['M6+', 'M6+ istor', 'M6+ instr']
        return self.eq_all, self.eq_ist, self.eq_inst, legend

    def set_save_path(self, folder_name, res):
        if folder_name == '':
            save_folder_name = '%s_P=%s' % (res.alg_name, res.lenf)
        else:
            save_folder_name = folder_name

        if os.name == 'nt':
            self.save_path = 'C:\\Users\\smoky\\Documents\\workspace\\result\\Barrier\\%s\\' % save_folder_name
        elif os.name == 'posix':
            self.save_path = '/Users/Ivan/Documents/workspace/result/Barrier/%s/' % save_folder_name
        else:
            print('os not supported')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
