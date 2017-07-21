import os
import numpy as np
from bmain.alghTools.tools import read_csv


class ImportData:
    def __init__(self, zone='', ln_field=False, gridVers=False, folder_name=''):
        self.zone = zone
        self.folder_name = folder_name
        self.res_path = os.path.expanduser('~' + os.getenv("USER") + '/Documents/workspace/resources/csv/Barrier/%s/'%zone)

        if gridVers:
            self.res_path += 'gridVers/d0.1/'

        self.col = self._read_txt_param(self.res_path)

        self.data_full = read_csv(self.res_path + 'khar.csv', self.col).T

        if ln_field:
            self.data_field = read_csv(self.res_path + 'field.csv', self.col).T
        else:
            self.data_field = read_csv(self.res_path + 'khar.csv', self.col).T

        self.data_sample = read_csv(self.res_path + 'sample.csv', self.col).T
        self.data_coord = read_csv(self.res_path + 'coord.csv', ['x', 'y']).T

        file_name_ist = '_eq_istor.csv'
        file_name_inst = '_eq_instr.csv'
        self.eq_ist = read_csv(self.res_path + file_name_ist, ['x', 'y']).T
        self.eq_inst = read_csv(self.res_path + file_name_inst, ['x', 'y']).T
        self.eq_all = np.append(self.eq_ist, self.eq_inst, axis=0)

    def get_eq_stack(self):
        M = 6
        legend = ['M%s+'%M, 'M%s+ istor'%M, 'M%s+ instr'%M]
        return self.eq_all, self.eq_ist, self.eq_inst, legend

    def set_save_path(self, alg_name, lenf):
        if self.folder_name == '':
            save_folder_name = '%s_%s_P=%s' % (self.zone, alg_name, lenf)
        else:
            save_folder_name = self.folder_name

        self.save_path = os.path.expanduser('~' + os.getenv("USER") +
                                            '/Documents/workspace/result/Barrier/%s/' % save_folder_name)
        original_umask = os.umask(0)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        os.umask(original_umask)

    def get_sample_coords(self):
        idx_feat_sample = read_csv(self.res_path + 'sample.csv', self.col).T[:, 0]
        #idx_feat_sample = [self.data_sample[0][0]]

        idx_array = []
        for j, idx in enumerate(self.data_full[:, 0]):
            if idx in idx_feat_sample:
                idx_array.append(j)
        return self.data_coord[idx_array]

    def _read_txt_param(self, res_dir):
        path = res_dir + 'param_str.txt'
        f = open(path, 'r')
        f_str = f.read()
        for s in ['[', ']', "'", ',']:
            f_str = f_str.replace(s, '')
        f_list = f_str.split()
        return f_list

    def read_cora_res(self, c=1):
        """импорт результатов алгоритма EPA"""
        CORAres = read_csv(self.res_path+'/kvz_CORA_result.csv', ['idx', 'r1', 'r2', 'r3', 'r4']).T
        idxCX = self.data_full[:, 0]
        idxX = np.array([]).astype(int)
        for res in CORAres:
            if '+' in res[c]:
                idxRes = np.where(idxCX == res[0])[0][0]
                idxX = np.append(idxX, idxRes)
        return idxX