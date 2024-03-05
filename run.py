from option import get_args
from advrm import ADVRM
import pandas as pd
import time
from log import record_hparams, log_scale_eval
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    current=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    writer = SummaryWriter(log_dir = f"{args['log_dir']}/{current}{args['log_dir_comment']} " ,comment=args['log_dir_comment'])
    # record_hparams(args, writer)

    csv_file = args['csv_dir']
    scene_set =  pd.read_csv(csv_file)

    if len(args['patch_size'].split(','))==2:
        patch_size=args['patch_size'].split(',')
        patch_size = [int(i) for i in patch_size]
    else:
        patch_size = None
    
    idx = args['idx']
    if idx >=0:
        scene_file = scene_set['Filename'].to_list()[idx]
        secen_key_points = scene_set.iloc[idx]
        if not args['random_object_flag']:
            args['obj_full_mask_file'] = f'{scene_file[:-4]}_mask.png'
        advrm=ADVRM(args, writer, None, True, args['random_object_flag'])
        advrm.run(args['scene_dir'], scene_file, idx, secen_key_points)
    else:
        for idx in range(args['scene_num']):
            scene_file = scene_set['Filename'].to_list()[idx]
            secen_key_points = scene_set.iloc[idx]
            if not args['random_object_flag']:
                args['obj_full_mask_file'] = f'{scene_file[:-4]}_mask.png'
            advrm=ADVRM(args, writer, None, True, args['random_object_flag'])
            advrm.run(args['scene_dir'], scene_file, idx, secen_key_points)

if __name__=='__main__':
    args = get_args()
    torch.cuda.set_device(args['device'])
    setup_seed(args['seed'])
    main(args)