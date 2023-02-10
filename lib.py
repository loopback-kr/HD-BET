# Python Built-in libraries
import os, sys, json, csv, re, random, numpy as np
from os.path import join, basename, exists, splitext, dirname, isdir
from shutil import copy, copytree, rmtree
from glob import glob, iglob
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from multiprocessing.pool import Pool
import multiprocessing.pool as mpp


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
    with tqdm(total=len(args[0]), desc=desc, colour="green", dynamic_ncols=True) as pbar:
        with Pool(num_processes) as pool:
            for _ in pool.istarmap(func, zip(*args)):
                pbar.update()
            pool.close()
            pool.join()