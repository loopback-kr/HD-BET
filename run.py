# Python Built-in libraries
import os, numpy as np
from tqdm import tqdm
from os.path import join, basename, exists, splitext, dirname, isdir
from glob import glob, iglob
from multiprocessing.pool import Pool
import multiprocessing.pool as mpp
# HD-BET
import torch, HD_BET
from HD_BET.run import run_hd_bet
from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# required to download pretrained model at ~/hd-bet_params/0.model
# required to download pretrained model at ~/hd-bet_params/1.model
# required to download pretrained model at ~/hd-bet_params/2.model
# required to download pretrained model at ~/hd-bet_params/3.model
# required to download pretrained model at ~/hd-bet_params/4.model

INPUT_PATHSES = [
    sorted(glob('ATLASR20_T1W_Label/imagesTr/*.nii.gz')),
    sorted(glob('ATLASR20_T1W_Label/imagesTs/*.nii.gz')),
]
DST_DIRS = [
    'ATLASR20_skullstrip/imagesTr',
    'ATLASR20_skullstrip/imagesTs',
]
GPU_COUNT = torch.cuda.device_count()

def istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

def run_multiproc(func, *args, desc='', num_processes=os.cpu_count()):
    with tqdm(total=len(args[0][0]), desc=desc, colour="green", dynamic_ncols=True) as pbar:
        with Pool(num_processes) as pool:
            for _ in pool.istarmap(func, zip(*args)):
                pbar.update()
            pool.close()
            pool.join()

if __name__ == '__main__':
    for INPUT_PATHS, DST_DIR in zip(INPUT_PATHSES, DST_DIRS):
        os.makedirs(DST_DIR, exist_ok=True)
        output_fnames = [join(DST_DIR, basename(input_path)) for input_path in INPUT_PATHS]
        run_multiproc(
            run_hd_bet,
            [input_paths.tolist() for input_paths in np.array_split(INPUT_PATHS, GPU_COUNT)],
            [output_fnames.tolist() for output_fnames in np.array_split(output_fnames, GPU_COUNT)],
            ['accurate'] * GPU_COUNT,
            [join(HD_BET.__path__[0], "config.py")] * GPU_COUNT,
            list(range(GPU_COUNT)),
            [True] * GPU_COUNT,
            [True] * GPU_COUNT,
            [True] * GPU_COUNT,
            [True] * GPU_COUNT,
            desc='Skullstripping',
            num_processes=GPU_COUNT
        )
