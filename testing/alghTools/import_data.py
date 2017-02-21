import os
import numpy as np
from testing.alghTools.tools import read_csv


class ImportData:
    def __init__(self, zone='', gridVers=False):
        if os.name == 'nt':
            res_dir = 'C:\\Users\\smoky\\Documents\workspace\\resources\\csv\\Barrier\\'
            eq_dir = 'C:\\Users\\smoky\\Documents\\workspace\\resources\\csv\\geop\\kvz\\'
        elif os.name == 'posix':
            eq_dir = '/Users/Ivan/Documents/workspace/resources/csv/geop/%s/' % zone
            if gridVers:
                zone = zone + '/gridVers/d0.2/'
            res_dir = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/%s/' % zone

        else:
            print('os not supported')

        if gridVers:
            """col = ['idx', 'Hmax', 'Hmin', 'DH', 'DH/l', 'Hdisp', 'Bmax', 'Bmin', 'DB', 'Bdisp', 'Mmax',
                   'Mmin', 'DM', 'Mdisp']"""
            col = ['idx', 'Hmax', 'Hmin', 'DH', 'DH/l', 'Bmax', 'Bmin', 'DB', 'Mmax',
                   'Mmin', 'DM',]
        else:
            """col = ['idx', 'Hmax', 'Hmin', 'DH', 'Top', 'Q', 'HR', 'Nl', 'Rint', 'DH/l', 'Nlc', 'R1', 'R2',
               'Bmax', 'Bmin', 'DB', 'Mmax', 'Mmin', 'DM', 'dps', 'Hdisp', 'Bdisp']"""
            col = ['idx', 'Hmax', 'Hmin', 'DH', 'DH/l', 'Bmax', 'Bmin', 'DB', 'Mmax', 'Mmin', 'DM']

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
