from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

def log_img_train(log, epoch, name_prefix, images):
    adv_scene_image, ben_scene_image, patch, adv_depth, ben_depth = images

    colormap = plt.get_cmap('viridis')
    ben_depth = ben_depth.detach().cpu()[0]
    adv_depth = adv_depth.detach().cpu()[0]
    ben_depth = (ben_depth - ben_depth.min()) / (ben_depth.max() - ben_depth.min())
    adv_depth = (adv_depth - adv_depth.min()) / (adv_depth.max() - adv_depth.min())
    ben_depth = colormap(ben_depth.numpy())[..., :3]
    adv_depth = colormap(adv_depth.numpy())[..., :3]
  
    log.add_image(f'{name_prefix}/train/adv_scene', adv_scene_image.detach().cpu()[0], epoch)
    log.add_image(f'{name_prefix}/train/ben_scene', ben_scene_image.detach().cpu()[0], epoch)
    log.add_image(f'{name_prefix}/train/patch', patch.detach().cpu()[0], epoch)
    log.add_image(f'{name_prefix}/train/ben_depth', ben_depth, epoch, dataformats='HWC')
    log.add_image(f'{name_prefix}/train/adv_depth', adv_depth, epoch, dataformats='HWC')


def log_scale_train(log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr):
    log.add_scalar(f'{name_prefix}/train/style_score', style_score.item(), epoch)
    log.add_scalar(f'{name_prefix}/train/content_score', content_score.item(), epoch)
    log.add_scalar(f'{name_prefix}/train/tv_score', tv_score.item(), epoch)
    log.add_scalar(f'{name_prefix}/train/adv_loss', adv_loss.item(), epoch)
    log.add_scalar(f'{name_prefix}/train/mean_shift', mean_shift.item(), epoch)
    log.add_scalar(f'{name_prefix}/train/max_shift', max_shift.item(), epoch)
    log.add_scalar(f'{name_prefix}/train/min_shift', min_shift.item(), epoch)
    log.add_scalar(f'{name_prefix}/train/mrsr', (mean_shift/(mean_ori+1e-7)).item(), epoch)
    log.add_scalar(f'{name_prefix}/train/arr', arr.item(), epoch)

def log_scale_eval(log,epoch, name_prefix, model_name, category, mean_shift, max_shift, min_shift, ror, arr):
    log.add_scalar(f'{name_prefix}/eval/{model_name}/{category}/mean_shift', mean_shift, epoch)
    log.add_scalar(f'{name_prefix}/eval/{model_name}/{category}/max_shift', max_shift, epoch)
    log.add_scalar(f'{name_prefix}/eval/{model_name}/{category}/min_shift', min_shift, epoch)
    log.add_scalar(f'{name_prefix}/eval/{model_name}/{category}/mrsr', ror, epoch)
    log.add_scalar(f'{name_prefix}/eval/{model_name}/{category}/arr', arr, epoch)

def record_hparams(args, writer):
        # 为了避免繁杂，这里只记录需要调参的字段
    hparams = {
        'lambda': args['lambda'],
        'beta': args['beta'],
        'epoch': args['epoch'],
        'learning_rate': args['learning_rate'],
        'depth_model': args['depth_model'],
        'eot_train_flag': args['eot_train_flag'],
        'eot_test_flag': args['eot_test_flag'],
        'obj_type_train': args['obj_type_train'],
        'obj_type_test': args['obj_type_test'],
        'ae_method': args['ae_method'],
        'adv_loss_type': args['adv_loss_type'],
        'grad_type': args['grad_type'],
        'scene_num': args['scene_num'],
        'patch_height': args['patch_height'],
    }
    writer.add_hparams(hparam_dict=hparams, metric_dict={})

def load_log(path, epoch=500):
    # print(re.search(r'runs/(.*)/', path)[1])
    log_data = event_accumulator.EventAccumulator(path)
    log_data.Reload()
    
    analyse_log(log_data, epoch, basic=False, eval=True, obj_trans=False, mod_trans=False)

# 修改score[i]控制用于计算的epoch
def calculate_statistics(scores, epoch):
    idx = epoch // 100
    if epoch == 1000:
        idx = -1
    # print('epoch = ', epoch)
    # print('epoch = ', idx * 50 if idx > 0 else 550 + idx * 50)
    values = np.array([score[idx][1] for score in scores])
    if len(values) == 0:
        return 0,0,0
    return values.min().round(10), values.max().round(10), values.mean().round(10)

def print_statistics(data_dict, epoch):
    brief = []
    for metric, data in data_dict.items():
        min_val, max_val, mean_val = calculate_statistics(data, epoch)
        print(f'{metric:40}: Min={min_val:15}, Max={max_val:15}, Mean={mean_val:15}')
        brief.append(mean_val)
    for v in brief:
        print(v)

def analyse_log(log_data, epoch=500, basic=True, eval=True, obj_trans=False, mod_trans=False):
    basic_metrics = ['style_score', 'content_score', 'tv_score', 'adv_loss', 'train/mean_shift', 'train/max_shift',
               'train/min_shift', 'train/mrsr', 'train/arr']
    
    eval_metrics = ['eval/.*/mean_shift', 'eval/.*/max_shift', 'eval/.*/min_shift', 'eval/.*/pas/mrsr',  
                    'eval/.*/car/mrsr',  'eval/.*/obs/mrsr', 'eval/.*/pas/arr', 'eval/.*/car/arr', 'eval/.*/obs/arr']

    obj_trans_metrics = ['pas/mrsr', 'car/mrsr', 'obs/mrsr']

    # mod_trans_metrics = ['depthhints/mrsr', 'monodepth2/mrsr', 'manydepth/mrsr', 
    #                         'midas/mrsr', 'adabins/mrsr', 'glpn-kitti/mrsr',
    #                         'dpt-dinov2-small-kitti/mrsr', 'dpt-dinov2-base-kitti/mrsr']
    mod_trans_metrics = ['glpn-kitti/pas/mrsr', 'glpn-kitti/car/mrsr', 'glpn-kitti/obs/mrsr']
    
    flags = [basic, eval, obj_trans, mod_trans]
    metrics = [item for lst, flag in zip([basic_metrics, eval_metrics, obj_trans_metrics, mod_trans_metrics], flags) if flag for item in lst]
    data = {key: [] for key in metrics}

    process_keys(data, log_data)

    print_statistics(data, epoch)


def process_keys(data_list, log_data):
    for key in log_data.scalars.Keys():
        for key_pattern in data_list:
            # new_pattern = '6_.*_[0-9]/'+key_pattern
            if re.search(key_pattern, key):
                rs = log_data.scalars.Items(key)
                data_list[key_pattern].append([(i.step, i.value) for i in rs])
  

if __name__=='__main__':
    path = sys.argv[1]
    epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    load_log(path, epoch)
    # log_data = event_accumulator.EventAccumulator("/home/hangcheng/codes/MDE_Attack/AdvRM/log_tmp/2024-02-19-05-11-25 adabins bim all /1708319485.5764875")
    # log_data.Reload()