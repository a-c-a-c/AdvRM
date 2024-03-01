import os,sys
current_path = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(current_path)
sys.path.append(f'{project_root}/DeepPhotoStyle_pytorch')
sys.path.append(f'{project_root}/PSMNet')
sys.path.append(f'{project_root}/AdaBins')
sys.path.append(f'{project_root}/MiDaS')
sys.path.append(f'{project_root}/ZoeDepth')
sys.path.append('..')
from depth_model import import_depth_model
from model import get_style_model_and_losses
from utils import compute_lap
from lr_decay import PolynomialLRDecay
from PSMNet.models import *
from ACVNet.models.acv import *
from MiDaS.midas.model_loader import load_model as load_midas_model
from AdaBins.models import UnetAdaptiveBins
from AdaBins.model_io import load_checkpoint as load_adabins_checkpoint
from transformers import AutoImageProcessor, DPTForDepthEstimation
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from zoedepth.models.builder import build_model as build_zoe_model
from zoedepth.utils.config import get_config as get_zoe_config
from functools import partial

integrated_mde_models = [
    'depthhints',
    'monodepth2',
    'manydepth', 
    # 'psmnet', 
    # 'acvnet', 
    'midas', 
    'adabins',
    'glpn-kitti',
    'dpt-dinov2-small-kitti',
    'dpt-dinov2-base-kitti',
    'zoedepth'
]
def omi_grad(model, attn, fn):
    print('ignore attention gradient!!!')
    for name, m in model.named_modules():
        if attn in name:              
            m.register_backward_hook(fn)

def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
    # 检查是否正常调用
    # print('attention ignore is called!!!')
    mask = torch.ones_like(grad_in[0]) * gamma
    return (mask * grad_in[0][:], )

def attn_tgr(module, grad_in, grad_out, gamma):
    # print('attn hook')
    mask = torch.ones_like(grad_in[0]) * gamma
    out_grad = mask * grad_in[0][:]
    B,C,H,W = grad_in[0].shape
    out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
    max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
    max_all_H = max_all//W
    max_all_W = max_all%W
    min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
    min_all_H = min_all//W
    min_all_W = min_all%W


    out_grad[:,range(C),max_all_H,:] = 0.0
    out_grad[:,range(C),:,max_all_W] = 0.0
    out_grad[:,range(C),min_all_H,:] = 0.0 #
    out_grad[:,range(C),:,min_all_W] = 0.0
    return (out_grad, )


def q_tgr(module, grad_in, grad_out, gamma):
    # print('q hook')
    # cait Q only uses class token
    mask = torch.ones_like(grad_in[0]) * gamma
    out_grad = mask * grad_in[0][:]
    out_grad[:] = 0.0
    return (out_grad, )
    
def v_tgr(module, grad_in, grad_out, gamma):
    # print('v hook')
    mask = torch.ones_like(grad_in[0]) * gamma
    out_grad = mask * grad_in[0][:]
    c = grad_in[0].shape[1]
    out_grad_cpu = out_grad.data.clone().cpu().numpy()
    max_all = np.argmax(out_grad_cpu[:,:], axis = 0)
    min_all = np.argmin(out_grad_cpu[:,:], axis = 0)
        
    out_grad[max_all,range(c)] = 0.0
    out_grad[min_all,range(c)] = 0.0
    for i in range(len(grad_in)):
        if i == 0:
            return_dics = (out_grad,)
        else:
            return_dics = return_dics + (grad_in[i],)
    return return_dics

def mlp_tgr(module, grad_in, grad_out, gamma):
    # print('mlp hook')
    mask = torch.ones_like(grad_in[0]) * gamma
    out_grad = mask * grad_in[0][:]
    c = grad_in[0].shape[1]
    out_grad_cpu = out_grad.data.clone().cpu().numpy()
    max_all = np.argmax(out_grad_cpu[:,:], axis = 0)
    min_all = np.argmin(out_grad_cpu[:,:], axis = 0)
    out_grad[max_all,range(c)] = 0.0
    out_grad[min_all,range(c)] = 0.0
    for i in range(len(grad_in)):
        if i == 0:
            return_dics = (out_grad,)
        else:
            return_dics = return_dics + (grad_in[i],)
    return return_dics

def mlp_tgr_v2(module, grad_in, grad_out, gamma):
    mask = torch.ones_like(grad_in[0]) * gamma
    out_grad = mask * grad_in[0][:]
    c = grad_in[0].shape[2]
    out_grad_cpu = out_grad.data.clone().cpu().numpy()
    max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
    min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
    out_grad[:,max_all,range(c)] = 0.0
    out_grad[:,min_all,range(c)] = 0.0
    for i in range(len(grad_in)):
        if i == 0:
            return_dics = (out_grad,)
        else:
            return_dics = return_dics + (grad_in[i],)
    return return_dics

def load_target_model(model_name,args,cuda=None):
    device = cuda if cuda is not None else f'cuda:{args["device"]}'
    if model_name in ['depthhints','monodepth2','manydepth']:
        depth_model = import_depth_model((args['input_width'],args['input_height']), model_type=model_name,device=device).eval()
        depth_model = depth_model.cuda(device)
    
    # 新增模型
    elif model_name == 'midas':
        depth_model, _, _, _ = load_midas_model(
            torch.device(device),
            '/home/wh/nas/MDE_Attack/MiDaS/weights/dpt_beit_large_512.pt',
            'dpt_beit_large_512',
            False, 320, False
        )
        # depth_model.eval()
        depth_model = depth_model.to(device).eval()
        if 'omi' in args['grad_type']:
            print('hook attn_drop_mask_grad')
            drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
            omi_grad(depth_model, 'attn_drop', drop_hook_func)
        if 're' in args['grad_type']:
            attn_tgr_hook = partial(attn_tgr, gamma=0.25)
            v_tgr_hook = partial(v_tgr, gamma=0.75)
            mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)
            for name, m in depth_model.named_modules():
                if 'attn.attn_drop' in name:
                    m.register_backward_hook(attn_tgr_hook) 
                elif 'attn.qkv' in name:
                    m.register_backward_hook(v_tgr_hook)
                elif 'mlp' == name[-3:] and 'blocks' in name:
                    m.register_backward_hook(mlp_tgr_hook)



    elif model_name == 'adabins':
        MIN_DEPTH = 1e-3
        MAX_DEPTH_KITTI = 80
        N_BINS = 256
        model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
        pretrained_path = "/home/wh/nas/MDE_Attack/AdaBins/weights/AdaBins_kitti.pt"
        depth_model, _, _ = load_adabins_checkpoint(pretrained_path, model)
        depth_model = depth_model.to(device).eval()
    elif model_name == 'glpn-kitti':
        model_dir = "/home/wh/nas/MDE_Attack/hf-models/glpn-kitti"
        model = GLPNForDepthEstimation.from_pretrained(model_dir)
        depth_model = model.to(device).eval()
        if 'omi' in args['grad_type']:
            drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
            omi_grad(depth_model, 'attention.self.dropout', drop_hook_func)
        if 're' in args['grad_type']:
            attn_tgr_hook = partial(attn_tgr, gamma=0.25)
            v_tgr_hook = partial(v_tgr, gamma=0.75)
            q_tgr_hook = partial(q_tgr, gamma=0.75)
            mlp_tgr_hook = partial(mlp_tgr_v2, gamma=0.5)
            for name, m in depth_model.named_modules():
                if 'attention.self.dropout' in name:
                    m.register_backward_hook(attn_tgr_hook)
                elif 'attention.self.query' in name:
                    m.register_backward_hook(q_tgr_hook)
                elif 'attention.sefl.key' in name:
                    m.register_backward_hook(v_tgr_hook)
                elif 'attention.self.value' in name:
                    m.register_backward_hook(v_tgr_hook)
                elif 'mlp' == name[-3:] and 'block' in name:
                    m.register_backward_hook(mlp_tgr_hook)
    
    elif model_name == 'dpt-dinov2-small-kitti':
        model_dir = "/home/wh/nas/MDE_Attack/hf-models/dpt-dinov2-small-kitti"
        model = DPTForDepthEstimation.from_pretrained(model_dir)
        depth_model = model.to(device).eval()
        if 'omi' in args['grad_type']:
            drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
            omi_grad(depth_model, 'attention.attention.dropout', drop_hook_func)
        if 're' in args['grad_type']:
            attn_tgr_hook = partial(attn_tgr, gamma=0.25)
            v_tgr_hook = partial(v_tgr, gamma=0.75)
            q_tgr_hook = partial(q_tgr, gamma=0.75)
            mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

            for name, m in depth_model.named_modules():
                if 'attention.attention.dropout' in name:
                    m.register_backward_hook(attn_tgr_hook)
                elif 'attention.attention.query' in name:
                    m.register_backward_hook(q_tgr_hook)
                elif 'attention.attention.key' in name:
                    m.register_backward_hook(v_tgr_hook)
                elif 'attention.attention.value' in name:
                    m.register_backward_hook(v_tgr_hook)
                elif 'mlp' == name[-3:] and 'layer' in name:
                    m.register_backward_hook(mlp_tgr_hook)
    
    elif model_name == 'dpt-dinov2-base-kitti':
        model_dir = "/home/wh/nas/MDE_Attack/hf-models/dpt-dinov2-base-kitti"
        model = DPTForDepthEstimation.from_pretrained(model_dir)
        depth_model = model.to(device).eval()
        if 'omi' in args['grad_type']:
            # print('hook attn_drop_mask_grad')
            drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
            omi_grad(depth_model, 'attention.attention.dropout', drop_hook_func)
        if 're' in args['grad_type']:
            attn_tgr_hook = partial(attn_tgr, gamma=0.25)
            v_tgr_hook = partial(v_tgr, gamma=0.75)
            q_tgr_hook = partial(q_tgr, gamma=0.75)
            mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

            for name, m in depth_model.named_modules():
                if 'attention.attention.dropout' in name:
                    m.register_backward_hook(attn_tgr_hook)
                elif 'attention.attention.query' in name:
                    m.register_backward_hook(q_tgr_hook)
                elif 'attention.attention.key' in name:
                    m.register_backward_hook(v_tgr_hook)
                elif 'attention.attention.value' in name:
                    m.register_backward_hook(v_tgr_hook)
                elif 'mlp' == name[-3:] and 'layer' in name:
                    m.register_backward_hook(mlp_tgr_hook)
    elif model_name == 'zoedepth':
        # ZoeD_K
        conf = get_zoe_config("zoedepth", "infer", config_version="kitti")
        model_zoe_k = build_zoe_model(conf)
        depth_model = model_zoe_k.to(device).eval()
        if 'omi' in args['grad_type']:
            drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
            omi_grad(depth_model, 'attn.attn_drop', drop_hook_func)
    else:
        print(model_name,' is not a valid model')
        raise ValueError
    # elif model_name == 'psmnet':
    #     depth_model = stackhourglass(192)
    #     depth_model = nn.DataParallel(depth_model, device_ids=[args['device']])
    #     state_dict = torch.load(f'{project_root}/PSMNet/pretrained/pretrained_model_KITTI2015.tar')
    #     depth_model.load_state_dict(state_dict['state_dict'])
    #     depth_model = depth_model.to('cuda').eval()
    # elif model_name == 'acvnet':
    #     depth_model = ACVNet(192,False,False).to('cuda').eval()
    #     depth_model = nn.DataParallel(depth_model, device_ids=[args['device']])
    #     state_dict = torch.load(f'{project_root}/models/acvnet/sceneflow.ckpt')
    #     depth_model.load_state_dict(state_dict['model'])
    
    for param in depth_model.parameters():
        param.requires_grad = False

    return [model_name, depth_model]

def disp_to_depth(disp,min_depth,max_depth):
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp=min_disp+(max_disp-min_disp)*disp
    depth=1/scaled_disp
    return scaled_disp,depth

def predict_depth_fn(MDE,scene,detach=False):
    model_name,model = MDE
    if model_name in ['depthhints','monodepth2','manydepth']:
        depth_without_norm = model(scene)
        scaler=5.4
        depth=torch.clamp(disp_to_depth(torch.abs(depth_without_norm),0.1,80)[1]*scaler,max=80)
        # print(scene.size(), depth_without_norm.size(), depth.size())

    elif model_name == 'midas':
        depth_without_norm = model(scene)
        depth_min = depth_without_norm.min()
        depth_without_norm = depth_without_norm - depth_min
        depth_max = depth_without_norm.max()
        depth_without_norm = -1 * depth_without_norm
        depth_without_norm = depth_without_norm + depth_max
        depth = depth_without_norm / depth_max * 80

       
    
    elif model_name == 'adabins':
        bin_edges, predicted_depth = model(scene)
        # upsample to the same size
        depth_without_norm = torch.nn.functional.interpolate(
            predicted_depth,
            size=(320, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth=depth_without_norm
        # clip to valid values
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        depth[depth < MIN_DEPTH] = MIN_DEPTH
        depth[depth > MAX_DEPTH] = MAX_DEPTH
    
    elif model_name == 'glpn-kitti':
        depth_without_norm = model(pixel_values=scene).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth_without_norm.unsqueeze(1),
            size=(320, 1024),
            mode="bicubic",
            align_corners=False,
        )
        # print(scene.size(), depth_without_norm.size(), depth.size())
    
    elif model_name == 'dpt-dinov2-small-kitti':
        depth_without_norm = model(pixel_values=scene).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth_without_norm.unsqueeze(1),
            size=(320, 1024),
            mode="bicubic",
            align_corners=False,
        )
        # print(scene.size(), depth_without_norm.size(), depth.size())
    
    elif model_name == 'dpt-dinov2-base-kitti':
        depth_without_norm = model(pixel_values=scene).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth_without_norm.unsqueeze(1),
            size=(320, 1024),
            mode="bicubic",
            align_corners=False,
        )
    
    elif model_name == 'zoedepth':
        depth_without_norm = model(scene)['metric_depth']
        depth = torch.nn.functional.interpolate(
            depth_without_norm,
            size=(320, 1024),
            mode="bicubic",
            align_corners=False,
        )
        

    elif model_name in ['acvnet','psmnet']:
        # print(model.device, scene[0].device, scene[1].device)
        depth_without_norm = model(scene)[0] 
        depth=torch.clamp(0.54*721/(depth_without_norm),max=80).unsqueeze(0)
    
    else:
        print(f'!!! {model_name}')
        raise ValueError
    
    if detach:
        depth_without_norm=depth_without_norm.detach()
        depth=depth.detach() 
    
    return depth, depth_without_norm  
def predict_batch(batch,MDE):
    adv_scene_image_eot, ben_scene_image_eot, scene_img_eot, patch_full_mask, object_full_mask = batch
    model_name,_ = MDE
    if model_name in ['depthhints','monodepth2','manydepth', 'glpn-kitti', 'dpt-dinov2-small-kitti', 'dpt-dinov2-base-kitti', 'zoedepth']:
        
        depth_predicted,_ = predict_depth_fn( MDE, torch.cat([adv_scene_image_eot, ben_scene_image_eot, scene_img_eot], dim = 0))
        adv_depth, ben_depth, scene_depth = depth_predicted[0],depth_predicted[1],depth_predicted[2]
    else:
        adv_depth,_ = predict_depth_fn( MDE, adv_scene_image_eot)
        with torch.no_grad():
            ben_depth,_ = predict_depth_fn( MDE, ben_scene_image_eot)
            scene_depth,_ = predict_depth_fn( MDE, scene_img_eot)
        if model_name == 'adabins':
            adv_depth=adv_depth[0]
            ben_depth=ben_depth[0]
            scene_depth=scene_depth[0]
    tar_depth = scene_depth.clone().detach()
    batch_y=[adv_depth, ben_depth, scene_depth, tar_depth]
    return batch_y