# Python Built-in libraries
from os.path import join, basename, exists, splitext, dirname, isdir
from glob import glob, iglob
# HD-BET
import HD_BET
from HD_BET.run import run_hd_bet

# required to download pretrained model at ~/hd-bet_params/0.model
# required to download pretrained model at ~/hd-bet_params/1.model
# required to download pretrained model at ~/hd-bet_params/2.model
# required to download pretrained model at ~/hd-bet_params/3.model
# required to download pretrained model at ~/hd-bet_params/4.model

INPUT_PATHS = sorted(glob('ATLASR20_T1W_Label/imagesTr/*.nii.gz'))
DST_DIR = '_ATLASR20_skullstrip'

run_hd_bet(
    mri_fnames=INPUT_PATHS,
    output_fnames=[join(DST_DIR, basename(input_path)) for input_path in INPUT_PATHS],
    mode='accurate',
    config_file=join(HD_BET.__path__[0], "config.py"),
    device=0,
    postprocess=True,
    do_tta=True,
    keep_mask=True,
    overwrite=True
)