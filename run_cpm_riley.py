import os
import json
from datetime import date
from utils import read_matlab_mat
from multiprocessing import Pool, cpu_count
import numpy as np
from functools import partial
from cpm import run_cpm_thread

MAT_PATH = "G:/.shortcut-targets-by-id/1Y42MQjJzdev5CtNSh2pJh51BAqOrZiVX/IMAGEN/CPM_mat/"
JSON_FILES_PATH = "G:/.shortcut-targets-by-id/1Y42MQjJzdev5CtNSh2pJh51BAqOrZiVX/IMAGEN/CPM_json/"
JSON_FILE = "10fold_p_thresh_0.01_repeat10_iter0_behav_SSRT_timepoint_bsl_MID_268ROI_z0_mode_linear_ynorm_id.json"


PARALLEL_JOBS = int(cpu_count()/2)

if __name__ == "__main__":

    # read json file from argument. Give full path to json file.
    json_file = os.path.join(JSON_FILES_PATH, JSON_FILE)
    with open(json_file) as json_data:
        data = json.load(json_data)

    # read run settings and output run settings
    today_date = date.today().strftime('%Y-%m-%d')
    t = data['t']
    k = data['k']
    p_thresh = data['p_thresh']
    repeat = data['repeat']
    num_iter = data['num_iter']
    mat_path = MAT_PATH # data['mat_path']
    mat_name = data['mat_name']  # should contains behav_name, num_rois, num_contrasts, and y_norm
    zscore = data['zscore']
    mode = data['mode']
    y_norm = data['y_norm']
    base_dir = "./CPM_outputs" #data['base_dir']
    out_path = '{}/{}_{}fold_p_thresh_{}_repeat{}_iter{}_timepoint_{}_z{}_mode_{}_mat_{}'.format(
        base_dir, today_date, k, p_thresh, repeat, num_iter, t, int(zscore), mode, mat_name[:-4])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open('{}/run_settings.txt'.format(out_path), 'w') as f:
        f.write('Run Date: {}\n'.format(today_date))
        f.write('Json file: {}\n'.format(json_file))
        f.write('Time Point: {}\n'.format(t))
        f.write('Number of folds k: {}\n'.format(k))
        f.write('P threshold: {}\n'.format(p_thresh))
        f.write('Number of repeats: {}\n'.format(repeat))
        f.write('Number of iterations: {}\n'.format(num_iter))
        f.write('Path to mat: {}\n'.format(mat_path))
        f.write('mat name: {}\n'.format(mat_name))
        f.write('z-score training edges: {}\n'.format(int(zscore)))
        f.write('mode: {}\n'.format(mode))
        f.write('y norm method: {}\n'.format(y_norm))
        f.write('Output path: {}\n'.format(out_path))

    x, y, lst_subjectkey = read_matlab_mat(mat_path, mat_name)
    with open('{}/lst_subjkey_analyzed.txt'.format(out_path), 'w') as f:
        for subj in lst_subjectkey:
            f.write('{}\n'.format(subj))

    lst_of_i = []
    lst_of_yrun = []
    for i in range(0, repeat + num_iter):
        lst_of_i.append(i+1)
        if i < repeat:  # true behavioral data
            lst_of_yrun.append(y)
        else:
            y_run = np.random.permutation(y)
            lst_of_yrun.append(y_run)

    num_cpu = cpu_count()
    use_cpu = np.floor(num_cpu * 0.5).astype(int)
    if PARALLEL_JOBS > use_cpu:
        print("WARNING: using more than half of total CPU ({}). Change number of jobs to half of CPU ({}).".format(PARALLEL_JOBS, use_cpu))
        num_proc = use_cpu
    else:
        num_proc = PARALLEL_JOBS

    print("Using {} jobs".format(num_proc))
    pool = Pool(processes=num_proc)
    cpm_par = partial(run_cpm_thread, x=x, k=k, out_path=out_path, p_thresh=p_thresh, zscore=zscore, mode=mode)
    block = False
    if block:
        for i in range(len(lst_of_i)): # blocking version of pool.starmap so I can debug
            cpm_par(lst_of_yrun[i], lst_of_i[i])
    else:
        pool.starmap(cpm_par, zip(lst_of_yrun, lst_of_i))
    pool.close()
    pool.join()

    # the following code also works but sometimes have missing stdout
    # with Pool(processes=num_proc) as pool:
    #    cpm_par = partial(run_cpm_thread, x=x, k=k, out_path=out_path, p_thresh=p_thresh, zscore=zscore, mode=mode)
    #    pool.starmap(cpm_par, zip(lst_of_yrun, lst_of_i))
