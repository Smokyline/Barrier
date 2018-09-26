import os

import numpy as np

from module.tools import read_csv_pandas


class ImportData:
    def __init__(self, zone='', param=None, gridVers=False, folder_name=''):
        self.zone = zone
        self.gridVers = gridVers
        self.folder_name = folder_name
        self.res_path = os.path.expanduser('~' + os.getenv("USER") + '/Documents/workspace/resources/csv/Barrier/%s/'%zone)

        self.data_full, self.data_header = read_csv_pandas(self.res_path + 'khar.csv', header=True)

        self.data_header = self.data_header[3:][param.global_feats()]
        print(self.data_header)
        self.data = np.array(self.data_full[:, 3:])[:, param.global_feats()]
        self.data_coord = self.data_full[:, 1:3]

        self.train, self.train_coord = self.get_train_set_from_all(param.get_sample_ln_idx())


        self.eq_ist = read_csv_pandas(self.res_path + '_eq_istor.csv', header=False)[:, :2]
        self.eq_inst = read_csv_pandas(self.res_path + '_eq_instr.csv', header=False)[:, :2]
        self.eq_all = np.append(self.eq_ist, self.eq_inst, axis=0)

        self.EXT = None

    def get_eq_stack(self):
        M = 6
        legend = ['M%s+'%M, 'M%s+ istor'%M, 'M%s+ instr'%M]
        #legend = ['M%s+'%M, 'M5+', 'M6+']
        return self.eq_all, self.eq_ist, self.eq_inst, legend

    def set_save_path(self, alg_name='', lenf=''):
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

    def get_train_set_from_all(self, sample_ln_idx,):
        idx_array = []
        for j, idx in enumerate(self.data_full[:, 0]):
            if idx in sample_ln_idx:
                idx_array.append(j)
        return self.data[idx_array], self.data_full[idx_array, 1:3]

    def _read_txt_param(self, res_dir):
        path = res_dir + 'param_str.txt'
        f = open(path, 'r')
        f_str = f.read()
        for s in ['[', ']', "'", ',']:
            f_str = f_str.replace(s, '')
        f_list = f_str.split()
        return f_list

    def read_cora_res(self, c=1):
        #TODO изменить результаты коры для Кавказа
        """импорт результатов алгоритма EPA"""
        CORAres = read_csv_pandas(self.res_path+'kvz_CORA_result.csv', header=False)
        # ['idx', 'r1', 'r2', 'r3', 'r4']
        idxCX = self.data[:, 0]
        idxX = np.array([]).astype(int)
        for res in CORAres:
            if '+' in res[c]:
                idxRes = np.where(idxCX == res[0])[0][0]
                idxX = np.append(idxX, idxRes)
        return idxX

    def read_cora_res_2(self):
        #TODO изменить результаты коры для Кавказа
        """импорт результатов алгоритма EPA"""
        CORAres = read_csv_pandas(self.res_path+'/cora.csv', header=False)
        # ['idx', 'r1', 'r2', 'r3', 'r4']
        idx_all = self.data[:, 0]
        idx_B = np.array([]).astype(int)
        for i, res in enumerate(CORAres):
                idx_in_data = np.where(idx_all == res)[0][0]
                idx_B = np.append(idx_B, idx_in_data)
        return idx_B