import os
import numpy as np
from main.alghTools.tools import read_csv


class ImportData:
    def __init__(self, zone='', gridVers=False):
        self.gridVers = gridVers
        if os.name == 'nt':
            res_dir = 'C:\\Users\\smoky\\Documents\workspace\\resources\\csv\\Barrier\\'
            eq_dir = 'C:\\Users\\smoky\\Documents\\workspace\\resources\\csv\\geop\\kvz\\'
        elif os.name == 'posix':
            eq_dir = '/Users/Ivan/Documents/workspace/resources/csv/geop/kvz/'
            if self.gridVers:
                res_dir = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/gridVers/d0.1/'
            else:
                res_dir = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/kvz_upd/'
                #res_dir = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/%s/' % zone

        else:
            res_dir, eq_dir = None, None
            print('os not supported')
        if self.gridVers:
            self.col = self._read_txt_param(res_dir)
        else:
            self.col = self._read_txt_param(res_dir)

            #self.col = ['idx',	'Hmax',	'Hmin',	'DH',	'Top', 'Q',	'HR',	'Nl',	'Rint',	'DH/l',	'Nlc',	'R1',	'R2',
            #            'Bmax',	'Bmin',	'DB', 'Mmax',	'Mmin',	'DM',]

        self.data_full = read_csv(res_dir + zone +'_khar.csv', self.col).T

        self.data_field = read_csv(res_dir + zone+'_khar.csv', self.col).T
        #self.data_field = read_csv(res_dir + 'kvz_field.csv', self.col).T

        self.data_sample = read_csv(res_dir + zone+'_sample.csv', self.col).T
        self.data_coord = read_csv(res_dir +zone+'_coord.csv', ['x', 'y']).T
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

    def set_save_path(self, folder_name, res=None):
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

    def _read_txt_param(self, res_dir):
        path = res_dir + 'param_str.txt'
        f = open(path, 'r')
        f_str = f.read()
        for s in ['[', ']', "'", ',']:
            f_str = f_str.replace(s, '')
        f_list = f_str.split()
        return f_list