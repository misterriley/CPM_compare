import pandas
import scipy.io as sio
import scipy.stats as stats
import os
import numpy as np
from pingouin import partial_corr

MAT_FILES_PATH = "G:/.shortcut-targets-by-id/1Y42MQjJzdev5CtNSh2pJh51BAqOrZiVX/IMAGEN/CPM_mat/"

if __name__ == '__main__':
    for file in os.listdir(MAT_FILES_PATH):

        if not file.endswith(".mat"):
            continue

        print(f"Loading {file}")
        mat_file = sio.loadmat(f"{MAT_FILES_PATH}{file}")

        subjectkeys = mat_file["subjectkey"]
        y = np.reshape(mat_file["y"], (-1))

        x = mat_file["x"]
        num_nodes = x.shape[1]
        idx = np.triu_indices(n=268, k=1) # make it a little easier on ourselves - cut out redundant parameters
        x = x[idx[0], idx[1],:]
        x = np.reshape(x, (len(y), -1))
        x_t = np.transpose(x)

        corrs = [stats.spearmanr(a, y) for a in x_t]
        mask = [1 if c[1] < .05 else 0 for c in corrs]
        x_masked = np.multiply(x, mask)