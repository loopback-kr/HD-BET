# Python Built-in libraries
import os, sys, json, csv, re, random, numpy as np
from os.path import join, basename, exists, splitext, dirname, isdir
from shutil import copy, copytree, rmtree
from glob import glob, iglob
from tqdm import tqdm, trange
from tqdm.contrib import tzip
# Medical imaging libraries
import nibabel as nib, SimpleITK as sitk
from lib import *

ORIGIN_LBL_DIR = 'ATLASR20_T1W_Label/labelsTr'
DST_LBL_DIR = 'ATLASR20_skullstrip_T1W_Label/labelsTr'  
MASK_DIR = 'ATLASR20_skullstrip_T1W_Label/imagesTr_skullstrip_mask'

old_label_dir = 'ATLASR20_skullstripped/labelsTr'
old_label_paths = sorted(glob(join(old_label_dir, '*.nii.gz')))



def align_label_after_skullstrip(src_path, msk_path, dst_path):
    src = nib.load(src_path)
    msk = nib.load(msk_path)
    aligned_lbl = np.multiply(src.get_fdata(), msk.get_fdata())
    aligned = nib.Nifti1Image(aligned_lbl, header=src.header, affine=src.affine) 
    nib.save(aligned, dst_path)


if __name__ == '__main__':
    if isdir(DST_LBL_DIR):
        raise Exception('There is a destination directory already.') 
    else:
        os.makedirs(DST_LBL_DIR)
    
    origin_lbl_paths = sorted(glob(join(ORIGIN_LBL_DIR, '*.nii.gz')))
    run_multiproc(
        align_label_after_skullstrip,
            origin_lbl_paths,
            sorted(glob(join(MASK_DIR, '*.nii.gz'))),
            [join(DST_LBL_DIR, basename(path)) for path in origin_lbl_paths],
        desc='Align labels',
    )
