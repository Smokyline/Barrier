import os
import numpy as np
from testing.alghTools.tools import read_csv


class ImportData:
    def __init__(self, zone='', gridVers=False):
        self.gridVers = gridVers
        if os.name == 'nt':
            res_dir = 'C:\\Users\\smoky\\Documents\workspace\\resources\\csv\\Barrier\\'
            eq_dir = 'C:\\Users\\smoky\\Documents\\workspace\\resources\\csv\\geop\\kvz\\'
        elif os.name == 'posix':
            eq_dir = '/Users/Ivan/Documents/workspace/resources/csv/geop/kvz/'
            if self.gridVers:
                res_dir = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/gridVers/d0.1cut/'
            else:
                res_dir = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/kvz_upd/'

        else:
            print('os not supported')
        if self.gridVers:
            self.col = ['idx', 'Hmax', 'Hmin', 'Hdelta', 'Hgrad', 'Hdisp', 'Hnch', 'Hcnt', 'Hw3-1', 'Hw3-2'
, 'Hw3-3', 'Hw3-1wei', 'Hw3-2wei', 'Hw3-3wei', 'Hw5-1', 'Hw5-2', 'Hw5-3', 'Hw5-4'
, 'Hw5-5', 'Hw5-1wei', 'Hw5-2wei', 'Hw5-3wei', 'Hw5-4wei', 'Hw5-5wei', 'Bmax'
, 'Bmin', 'Bdelta', 'Bgrad', 'Bdisp', 'Bnch', 'Bcnt', 'Bw3-1', 'Bw3-2', 'Bw3-3'
, 'Bw3-1wei', 'Bw3-2wei', 'Bw3-3wei', 'Bw5-1', 'Bw5-2', 'Bw5-3', 'Bw5-4', 'Bw5-5'
, 'Bw5-1wei', 'Bw5-2wei', 'Bw5-3wei', 'Bw5-4wei', 'Bw5-5wei', 'Mmax', 'Mmin'
, 'Mdelta', 'Mgrad', 'Mdisp', 'Mnch', 'Mcnt', 'Mw3-1', 'Mw3-2', 'Mw3-3', 'Mw3-1wei'
, 'Mw3-2wei', 'Mw3-3wei', 'Mw5-1', 'Mw5-2', 'Mw5-3', 'Mw5-4', 'Mw5-5', 'Mw5-1wei'
, 'Mw5-2wei', 'Mw5-3wei', 'Mw5-4wei', 'Mw5-5wei', 'Amax', 'Amin', 'Adelta', 'Agrad'
, 'Adisp', 'Anch', 'Acnt', 'Aw3-1', 'Aw3-2', 'Aw3-3', 'Aw3-1wei', 'Aw3-2wei'
, 'Aw3-3wei', 'Aw5-1', 'Aw5-2', 'Aw5-3', 'Aw5-4', 'Aw5-5', 'Aw5-1wei', 'Aw5-2wei'
, 'Aw5-3wei', 'Aw5-4wei', 'Aw5-5wei']
        else:

            self.col = ['idx', 'Hmax', 'Hmin', 'Hdelta', 'Hgrad', 'Hdisp', 'Hnch', 'Hcnt', 'Hw3-1', 'Hw3-2'
                , 'Hw3-3', 'Hw3-1wei', 'Hw3-2wei', 'Hw3-3wei', 'Hw5-1', 'Hw5-2', 'Hw5-3', 'Hw5-4'
                , 'Hw5-5', 'Hw5-1wei', 'Hw5-2wei', 'Hw5-3wei', 'Hw5-4wei', 'Hw5-5wei', 'Bmax'
                , 'Bmin', 'Bdelta', 'Bgrad', 'Bdisp', 'Bnch', 'Bcnt', 'Bw3-1', 'Bw3-2', 'Bw3-3'
                , 'Bw3-1wei', 'Bw3-2wei', 'Bw3-3wei', 'Bw5-1', 'Bw5-2', 'Bw5-3', 'Bw5-4', 'Bw5-5'
                , 'Bw5-1wei', 'Bw5-2wei', 'Bw5-3wei', 'Bw5-4wei', 'Bw5-5wei', 'Mmax', 'Mmin'
                , 'Mdelta', 'Mgrad', 'Mdisp', 'Mnch', 'Mcnt', 'Mw3-1', 'Mw3-2', 'Mw3-3', 'Mw3-1wei'
                , 'Mw3-2wei', 'Mw3-3wei', 'Mw5-1', 'Mw5-2', 'Mw5-3', 'Mw5-4', 'Mw5-5', 'Mw5-1wei'
                , 'Mw5-2wei', 'Mw5-3wei', 'Mw5-4wei', 'Mw5-5wei', 'Amax', 'Amin', 'Adelta', 'Agrad'
                , 'Adisp', 'Anch', 'Acnt', 'Aw3-1', 'Aw3-2', 'Aw3-3', 'Aw3-1wei', 'Aw3-2wei'
                , 'Aw3-3wei', 'Aw5-1', 'Aw5-2', 'Aw5-3', 'Aw5-4', 'Aw5-5', 'Aw5-1wei', 'Aw5-2wei'
                , 'Aw5-3wei', 'Aw5-4wei', 'Aw5-5wei']
            #self.col = ['idx',	'Hmax',	'Hmin',	'DH',	'Top', 'Q',	'HR',	'Nl',	'Rint',	'DH/l',	'Nlc',	'R1',	'R2',
             #           'Bmax',	'Bmin',	'DB', 'Mmax',	'Mmin',	'DM',	'dps',	'Hdisp',	'Bdisp']

        self.data_full = read_csv(res_dir + 'kvz_khar.csv', self.col).T

        self.data_field = read_csv(res_dir + 'kvz_khar.csv', self.col).T
        #self.data_field = read_csv(res_dir + 'kvz_field.csv', self.col).T

        self.data_sample = read_csv(res_dir + 'kvz_sample.csv', self.col).T
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
