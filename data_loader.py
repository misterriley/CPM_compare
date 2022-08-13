import os

import mat73
import numpy as np
import pandas as pd
import scipy.io as sio

MAT_FILES_DICT = \
    {
        "sadie-marie":
            {
                "path": "G:\\.shortcut-targets-by-id\\1P67X2oPl5kWND4p5dhdn9RbtEZcHaiZ9"
                        "\\Data CPM 2460 Accuracy Interference",
                "file":
                    [
                        "rest_estroop_acc_interf_2460_cpm_ready.mat",
                        "enback_estroop_acc_interf_2460_cpm_ready.mat"
                    ],
                "y_in_mat_file": True,
                "x_col": "x",
                "y_col": "y"
            },
        "test_data":
            {
                "path": "G:\\My Drive\\CPM_test_data",
                "file": "stp_all_clean2.mat",
                "y_in_mat_file": False,
                "x_col": "stp_all",
                "y_file": "G:\\My Drive\\CPM_test_data\\txnegop.txt"
            },
        "IMAGEN":
            {
                "path": "G:\\.shortcut-targets-by-id\\1Nj5b1RhD0TcXoswrxiV5gkSPPq4fnGfu\\IMAGEN_master_data"
                        "\\Matrices_Qinghao_new\\matrices",
                "file":
                    {
                        "mats_mid_bsl.mat": {'x_col': 'mats_mid'},
                        "mats_mid_fu2.mat": {'x_col': 'mats_mid'},
                        "mats_sst_bsl.mat": {'x_col': 'mats_sst'},
                        "mats_sst_fu2.mat": {'x_col': 'mats_sst'},
                    },
                "y_in_mat_file": False,
                "y_file": "G:\\.shortcut-targets-by-id\\1Y42MQjJzdev5CtNSh2pJh51BAqOrZiVX\\IMAGEN\\behav_variables"
                          "\\behav_variables_only.xlsx",
                "y_col":
                    [
                        "kirby_c_estimated_k_all_trials_fu2",
                        "csi_c_sum_fu2",
                        "neo_c_neuroticism_average_fu2",
                        "neo_c_extraversion_average_fu2",
                        "neo_c_openness_average_fu2",
                        "neo_c_agreeableness_average_fu2",
                        "neo_c_conscientiousness_average_fu2",
                        "surps_c_hopelessness_average_fu2",
                        "surps_c_anxiety_sensitivity_average_fu2",
                        "surps_c_impulsivity_average_fu2",
                        "surps_c_sensation_seeking_average_fu2"
                    ]
            }
    }


class DataSet:

    def __init__(self, descriptor, x, y):
        if isinstance(x, pd.DataFrame):
            self.x = x.values
        else:
            self.x = x

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            self.y = y.values
        else:
            self.y = y
        self.y = self.y.reshape(-1)

        self.descriptor = descriptor

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_descriptor(self):
        return self.descriptor


def load_txt(file, delimiter=","):
    y_ = pd.read_csv(file, sep=delimiter, header=None)
    y_ = y_.values.reshape(-1)
    return y_


def load_mat(file):
    print("Loading {}".format(file))
    try:
        return sio.loadmat(file)
    except NotImplementedError:
        return mat73.loadmat(file)


class DataLoader:

    def __init__(self, protocol_c=None, file_c=None, y_col_c=None):
        self.data_sets = []

        self.display_protocol = None
        self.display_file = None
        self.display_y_col = None

        self.protocol_c = protocol_c
        if isinstance(self.protocol_c, str):
            self.protocol_c = [self.protocol_c]
        self.file_c = file_c
        if isinstance(self.file_c, str):
            self.file_c = [self.file_c]
        self.y_col_c = y_col_c
        if isinstance(self.y_col_c, str):
            self.y_col_c = [self.y_col_c]
        self.load_data()

    def build_data_set(self, x, y):
        descriptor = self.display_protocol + \
                     (("_" + self.display_file.split(".")[0]) if self.display_file is not None else "") + \
                     (("_" + self.display_y_col) if self.display_y_col is not None else "")
        self.data_sets.append(DataSet(descriptor, x, y))

    def get_data_sets(self):
        return self.data_sets

    def protocol_allowed(self, protocol):
        return self.protocol_c is None or protocol in self.protocol_c

    def file_allowed(self, file):
        return self.file_c is None or file in self.file_c

    def y_col_allowed(self, y_col):
        return self.y_col_c is None or y_col in self.y_col_c

    def load_data(self):
        for protocol_m in MAT_FILES_DICT:
            self.display_protocol = protocol_m
            if self.protocol_allowed(protocol_m):
                mat_file_m = MAT_FILES_DICT[protocol_m]["file"]
                if isinstance(mat_file_m, str) and self.file_allowed(mat_file_m):
                    self.load_mat_file(protocol_m, mat_file_m)
                elif isinstance(mat_file_m, list):
                    for file_m in mat_file_m:
                        self.display_file = file_m
                        if self.file_allowed(file_m):
                            self.load_mat_file(protocol_m, file_m)
                    self.display_file = None
                elif isinstance(mat_file_m, dict):
                    for file_m in mat_file_m:
                        self.display_file = file_m
                        if self.file_allowed(file_m):
                            self.load_mat_file(protocol_m, file_m, mat_file_m[file_m])
                    self.display_file = None
        self.display_protocol = None

    def add_one_y_col(self, y_col, x, mat):
        if self.y_col_allowed(y_col):
            y = mat[y_col]
            self.build_data_set(x, y)

    def add_y_col_from_mat(self, y_col, x, mat):
        if isinstance(y_col, str):
            self.add_one_y_col(y_col, x, mat)
        else:
            for yc in y_col:
                self.display_y_col = yc
                self.add_one_y_col(yc, x, mat)
            self.display_y_col = None

    def load_mat_file(self, protocol, file, data=None):
        path_m = MAT_FILES_DICT[protocol]["path"]
        y_in_mat_file_m = MAT_FILES_DICT[protocol]["y_in_mat_file"]
        x_col_m = MAT_FILES_DICT[protocol].get("x_col")
        if x_col_m is None:
            if data is None:
                raise ValueError("x_col is None and data is None")
            x_col_m = data["x_col"]
        y_col_m = MAT_FILES_DICT[protocol].get("y_col")
        y_file_m: str = MAT_FILES_DICT[protocol].get("y_file")
        mat = load_mat(os.path.join(path_m, file))
        x = mat[x_col_m]
        if y_in_mat_file_m:
            self.add_y_col_from_mat(y_col_m, x, mat)
        elif y_file_m.endswith(".txt"):
            y = load_txt(y_file_m, delimiter="\t")
            self.build_data_set(x, y)
        elif y_file_m.endswith(".xlsx"):
            y = pd.read_excel(io=y_file_m, dtype=np.float32)
            self.add_y_col_from_mat(y_col_m, x, y)
        else:
            mat = load_mat(y_file_m)
            self.add_y_col_from_mat(y_col_m, x, mat)


if __name__ == "__main__":

    dl = DataLoader(protocol_c=["IMAGEN"],
                    file_c=["mats_sst_fu2.mat"],
                    y_col_c=["surps_c_sensation_seeking_average_fu2"])
    data_sets = dl.get_data_sets()
    for data_set in data_sets:
        print(data_set.get_descriptor())
