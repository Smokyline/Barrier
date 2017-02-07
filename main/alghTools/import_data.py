import os
import numpy as np
from main.alghTools.tools import read_csv


class ImportData:
    def __init__(self, folder_name):
        if os.name == 'nt':
            res_dir = 'C:\\Users\\smoky\\Documents\workspace\\resourses\\csv\\newEPA\\'
            eq_dir = 'C:\\Users\\smoky\\Documents\\workspace\\resourses\\csv\\geop\\kvz\\'
            self.save_path = 'C:\\Users\\smoky\\Documents\\workspace\\result\\Barrier\\best\\'
        elif os.name == 'posix':
            res_dir = '/Users/Ivan/Documents/workspace/resourses/csv/newEPA/'
            eq_dir = '/Users/Ivan/Documents/workspace/resourses/csv/geop/kvz/'
            self.save_path = '/Users/Ivan/Documents/workspace/result/Barrier/%s/' % folder_name
        else:
            print('os not supported')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        col = ['idx', 'Hmax', 'Hmin', 'DH', 'Top', 'Q', 'HR', 'Nl', 'Rint', 'DH/l', 'Nlc', 'R1', 'R2',
               'Bmax', 'Bmin', 'DB', 'Mmax', 'Mmin', 'DM', 'dps', 'Hdisp', 'Bdisp']

        self.data_full = read_csv(res_dir + 'Caucasus_khar.csv', col).T
        self.data_sample = read_csv(res_dir + 'Caucasus_sample.csv', col).T
        self.data_coord = read_csv(res_dir + 'Caucasus_coord.csv', ['x', 'y']).T
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
