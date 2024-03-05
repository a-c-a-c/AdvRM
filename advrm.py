from images import *
from load_model import load_target_model, integrated_mde_models, predict_batch, predict_depth_fn
import pandas as pd
import torchvision.models as models
from model import get_style_model_and_losses
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from log import log_img_train,log_scale_train,log_scale_eval
import urllib
import traceback
from torch.optim import Adam
import torch.optim as optim
from lr_decay import PolynomialLRDecay
def bim(grad, input_patch, lr):  
    input_patch=(torch.clamp(input_patch-lr*grad.sign(),0,1)).detach() 
    input_patch.requires_grad_()
    return input_patch

def mifgsm(grad, mo, input_patch, lr):
    raise NotImplementedError

class ADVRM:
    def __init__(self,args, log, patch_size=None, rewrite=False, random_object_flag=True) -> None:
        self.args = args
        self.log = log
        self.patch = Patch(self.args)
        self.patch.load_patch_warp(self.args['patch_file'], self.args['patch_dir'], patch_size, rewrite = rewrite)
        self.objects = OBJ(self.args)
        if random_object_flag:
            self.objects.load_obj_warp(['pas','car','obs'])
            print('==================objects for training============')
            print(self.objects.object_files_train)
            print('==================objects for test============')
            print(self.objects.object_files_test)
            self.run = self.run_with_random_object
        else:
            # self.objects.load_object_mask(self.args['obj_full_mask_file'], self.args['obj_full_mask_dir'])
            self.run = self.run_with_fixed_object
        self.MDE = load_target_model(self.args['depth_model'], self.args)
        self.configure_loss(self.patch)

    def configure_loss(self, patch):
        def adv_loss_fn(batch, batch_y):
            adv_score = 0
            _, _, _, patch_full_mask, object_full_mask = batch
            adv_depth, ben_depth, _, tar_depth = batch_y
            object_depth = adv_depth * object_full_mask
            object_diff_ben = (adv_depth-ben_depth)*object_full_mask
            patch_full_mask = torch.clip(patch_full_mask - object_full_mask, 0, 1)
            patch_diff_tar= torch.abs((adv_depth-ben_depth)*patch_full_mask)
            adv_score += -object_depth.sum()/(object_full_mask.sum()+1e-7) \
                - object_depth[object_diff_ben<0].sum()/(object_full_mask[object_diff_ben<0].sum()+1e-7) \
                + patch_diff_tar.sum() / (patch_full_mask.sum()+1e-7)       
            return adv_score
        def patch_style_core(style_image, content_image, style_mask, content_mask, laplacian_m):
            cnn = models.vgg19(weights='DEFAULT').features.cuda(self.args['device']).eval()
            cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda(self.args['device'])
            cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).cuda(self.args['device'])
            style_model, style_losses, content_losses, tv_losses = get_style_model_and_losses(cnn,
                cnn_normalization_mean, cnn_normalization_std, style_image, content_image, style_mask, content_mask, laplacian_m)

            for param in style_model.parameters():
                param.requires_grad = False
            return style_model, style_losses, content_losses, tv_losses
        def total_variation_loss(img):
            bs_img, c_img, h_img, w_img = img.size()
            tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
            tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
            return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
        self.style_model, self.style_losses, self.content_losses, _  = patch_style_core(patch.patch_img_s, patch.patch_img_c,torch.torch.ones([1,patch.patch_img_s.shape[2],patch.patch_img_s.shape[3]]).cuda(self.args['device']), torch.torch.ones([1,patch.patch_img_s.shape[2],patch.patch_img_s.shape[3]]).cuda(self.args['device']), patch.laplacian_m)
        self.tv_loss_fn = total_variation_loss
        self.adv_loss_fn = adv_loss_fn
       
    def compute_sty_loss(self, patch, gt, style_weight, content_weight, tv_weight):
        style_score = torch.zeros(1).float().cuda(self.args['device'])
        content_score = torch.zeros(1).float().cuda(self.args['device'])
        tv_score = torch.zeros(1).float().cuda(self.args['device']) 
        self.style_model(patch)
        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss
        tv_score+=self.tv_loss_fn(patch-gt)
        style_loss = style_weight * style_score + content_weight * content_score + tv_weight * tv_score 
        return style_loss, style_score, content_score, tv_score

    def run_with_random_object(self, scene_dir, scene_file, idx, points):
        self.env = ENV(self.args, scene_file, scene_dir, idx, points)
        name_prefix = f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
        if self.args['update']=='lbfgs':
            optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
            LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            def closure(): 
                self.patch.optmized_patch.data.clamp_(0, 1)
                batch, patch_size = self.env.accept_patch_and_objects(False, self.patch.optmized_patch, self.patch.mask, self.objects.object_imgs_train, self.env.insert_range, None, None, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'], offset_object=self.args['train_offset_object_flag'], color_object=self.args['train_color_object_flag'])

                adv_scene_image, ben_scene_image, scene_img, patch_full_mask, object_full_mask = batch
                                                                

                # if self.args['up']==self.args['bottom']:
                #     try:
                #         assert patch_size_large[0] == patch_size[0]
                #         assert patch_size_large[1] == patch_size[1]
                #     except:
                #         print(f"assert error!!! given patch size:{patch_size_large}, calculated patch size:{patch_size}")
                #         exit()

                batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]

                batch_y = predict_batch(batch, self.MDE)
                
                adv_loss = self.adv_loss_fn(batch, batch_y)
                mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
                
                adv_loss = self.adv_loss_fn(batch, batch_y)
                style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
                loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
                if self.args['update']=='lbfgs':
                    loss.backward()

                if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
                    optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
                    self.patch.optmized_patch.data = optmized_patch.data
                
                if self.args['train_log_flag']:
                    if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                        log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1]])
                        self.log.add_image(f'{name_prefix}/train/object_mask', object_full_mask.detach().cpu()[0, 0], epoch, dataformats='HW')
                        
                        log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
                
                if self.args['inner_eval_flag']:
                    if epoch % self.args['inner_eval_interval']==0 or (epoch+1)==self.args['epoch']: 
                        for category in self.objects.object_imgs_test.keys():
                            if self.args['random_test_flag']:
                                record = [[] for i in range(5)]
                                for _ in range(20):
                                    record_tmp = self.eval(self.MDE, category)
                                    for i in range(5):
                                        record[i]+=record_tmp[i]       
                            else:
                                record = self.eval(self.MDE, category)
                            
                            log_scale_eval(self.log, epoch, name_prefix, self.MDE[0],category, np.mean(record[0]), np.mean(record[1]), np.mean(record[2]),np.mean(record[3]),np.mean(record[4]) )      
                # if (epoch+1)==self.args['epoch'] or (epoch % self.args['inner_eval_interval']==0 and epoch>300): 
        
                #         for idx in range(len(self.objects.object_imgs_test['pas'])-3+1):
                #             batch, _ = self.env.accept_patch_and_objects(True, self.patch.optmized_patch, self.patch.mask, self.objects.object_imgs_test, self.env.insert_range, None, None, offset_patch=self.args['test_offset_patch_flag'], color_patch=self.args['test_color_patch_flag'], offset_object=self.args['test_offset_object_flag'], color_object=self.args['test_color_object_flag'],object_idx_g=idx, category='pas')    
                #             batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
                #             batch_y= predict_batch(batch, self.MDE)

                #             adv_scene_image, ben_scene_image, scene_img, patch_full_mask, object_full_mask = batch
                #             adv_depth, ben_depth, scene_depth, tar_depth = batch_y

                #             colormap = plt.get_cmap('viridis')
                #             ben_depth = ben_depth.detach().cpu()[0]
                #             adv_depth = adv_depth.detach().cpu()[0]
                #             ben_depth = (ben_depth - ben_depth.min()) / (ben_depth.max() - ben_depth.min())
                #             adv_depth = (adv_depth - adv_depth.min()) / (adv_depth.max() - adv_depth.min())
                #             ben_depth = colormap(ben_depth.numpy())[..., :3]
                #             adv_depth = colormap(adv_depth.numpy())[..., :3]
                        
                #             self.log.add_image(f'{name_prefix}/eval/adv_scene_{idx}', adv_scene_image.detach().cpu()[0], epoch)
                #             self.log.add_image(f'{name_prefix}/eval/ben_scene_{idx}', ben_scene_image.detach().cpu()[0], epoch)
                #             self.log.add_image(f'{name_prefix}/eval/ben_depth_{idx}', ben_depth, epoch, dataformats='HWC')
                #             self.log.add_image(f'{name_prefix}/eval/adv_depth_{idx}', adv_depth, epoch, dataformats='HWC')

                # if (epoch+1)==self.args['epoch']:
                #     print(f'mean_ori:{mean_ori.item():.2f}\tmean_shift:{mean_shift.item():.2f}\tmrsr:{(mean_shift/mean_ori).item():.2f}') 
                return loss
            if self.args['update']=='lbfgs':
                optimizer.zero_grad()
                optimizer.step(closure)
                LR_decay.step()
            else:
                loss = closure()
                grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
                self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])

    def run_with_fixed_object(self, scene_dir, scene_file, idx, points):
        self.env = ENV(self.args, scene_file, scene_dir, idx, points)
        name_prefix = f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
        if self.args['update']=='lbfgs':
            optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
            LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            def closure(): 
                if self.args['update']=='lbfgs':
                    optimizer.zero_grad()
                self.patch.optmized_patch.data.clamp_(0, 1)
                adv_scene_image, ben_scene_image, patch_size, patch_full_mask = self.env.accept_patch(self.patch.optmized_patch, None, self.patch.mask, self.args['insert_height'], offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'])

                batch= [adv_scene_image, ben_scene_image, self.env.env, patch_full_mask, self.objects.object_full_mask]
                batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]

                batch_y = predict_batch(batch, self.MDE)
                
                adv_loss = self.adv_loss_fn(batch, batch_y)
                mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
                
                adv_loss = self.adv_loss_fn(batch, batch_y)
                style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
                loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
                if self.args['update']=='lbfgs':
                    loss.backward()

                if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
                    optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
                    self.patch.optmized_patch.data = optmized_patch.data
                if self.args['train_log_flag']:
                    if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                        log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1]])
                        log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
                
                return loss
            if self.args['update']=='lbfgs':

                optimizer.step(closure)
                LR_decay.step()
            else:
                loss = closure()
                grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
                self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])
    
    def eval(self, MDE, category, insert_height=None, patch=None):
        if patch is None:
            patch = self.patch.optmized_patch
        if self.args['test_quan_patch_flag']:
            patch = (patch*255).int().float()/255.
        if category == 'pas':
             object_num=3
        else:
            object_num=1

        with torch.no_grad():
            record = [[] for _ in range(5)]
            for idx in range(len(self.objects.object_imgs_test[category])-object_num+1):
                batch, _ = self.env.accept_patch_and_objects(True, patch, self.patch.mask, self.objects.object_imgs_test, self.env.insert_range, insert_height, None, offset_patch=self.args['test_offset_patch_flag'], color_patch=self.args['test_color_patch_flag'], offset_object=self.args['test_offset_object_flag'], color_object=self.args['test_color_object_flag'],object_idx_g=idx, category=category)
                
                batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
                batch_y= predict_batch(batch, MDE)
                
                mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
                record[0].append(mean_shift.item())
                record[1].append(max_shift.item())
                record[2].append(min_shift.item())
                record[3].append((mean_shift/mean_ori).item())
                record[4].append(arr.item())
            return record

    def eval_core(self, adv_depth, ref_depth, scene_obj_mask):
        shift=(adv_depth-ref_depth)*scene_obj_mask
        mean_shift = torch.sum(shift)/torch.sum(scene_obj_mask)
        mean_ori = torch.sum(ref_depth*scene_obj_mask)/torch.sum(scene_obj_mask)
        max_shift=shift[scene_obj_mask==1].max()
        min_shift=shift[scene_obj_mask==1].min()

        shift = (adv_depth - ref_depth)
        relative_shift = (shift / ref_depth)
        affect_region = relative_shift > 0.14
        affect_region = affect_region * scene_obj_mask[:, 0, :, :]
        arr = affect_region.sum() / scene_obj_mask[:, 0, :, :].sum()
        return mean_ori, mean_shift, max_shift, min_shift, arr


    # def run(self, scene_dir, scene_file, idx, points):
    #     eot_train = self.args['eot_train_flag'] 
    #     eot_test =  self.args['eot_test_flag'] 
    #     print(eot_train, eot_test)
        
    #     self.env = ENV(self.args, scene_file, idx, points)
    #     self.env.load_scene_warp(scene_file, scene_dir)
    #     name_prefix=f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
    #     merged_grad = 0

    #     for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            
    #         style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
            
    #         if self.args['grad_type'] in ['base', 'omi']:
    #             object_insert_param, star_idx, object_num, category = self.env.accept_help(self.objects.object_imgs_train,category=None, eot_object= eot_train)
    #             batch,_ = self.env.accept_patch_and_objects(self.patch.optmized_patch, 
    #             self.patch.init_patch, self.patch.mask, self.objects.object_imgs_train, self.env.insert_range, None,False, object_insert_param, star_idx, object_num, category, eot_patch=eot_train, eot_object=eot_train)
                
    #             batch_y = predict_batch(batch, self.MDE)
    #             adv_loss = self.adv_loss_fn(batch, batch_y)
    #             loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
    #             grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
    #         else:
    #             raise NotImplementedError

    #         merged_grad += grad
    #         if (epoch+1) % self.args['opt_step'] ==0:
    #             merged_grad /= self.args['opt_step']
    #             self.patch.optmized_patch = bim(merged_grad,self.patch.optmized_patch,self.args['learning_rate'])
    #             merged_grad = 0
    #         if eot_train and  (epoch+1)%10==0 and self.args['quan']:
    #             optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
    #             self.patch.optmized_patch.data = optmized_patch.data
    #             self.patch.optmized_patch.data.clamp_(0,1)

            
    #         #log 
    #         if self.args['train_log_flag']:
    #             if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 log_img_train(self.log, epoch, name_prefix, batch, batch_y)
    #             if epoch % self.args['train_scale_log_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
    #                 log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
    #                 print('logging',arr, mean_shift/mean_ori)


    #         if self.args['inner_eval_flag']:
    #             if epoch % self.args['inner_eval_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 for category in ['pas', 'car', 'obs']:
    #                     if eot_test:
    #                         if self.args['quan']:
    #                             optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
    #                         else:
    #                             optmized_patch = self.patch.optmized_patch

    #                         record = [[] for i in range(5)]
    #                         for _ in range(10):
    #                             record_tmp = self.eval(self.MDE, category, False, eot_patch=eot_test, eot_object=eot_test, patch=optmized_patch)
    #                             for i in range(5):
    #                                 record[i]+=record_tmp[i]
                            
    #                     else:
    #                         # if epoch in [300, 400, 499]:
    #                         #     record = self.eval(self.MDE, category, False, None, epoch, name_prefix)
    #                         # else:
    #                             record = self.eval(self.MDE, category, False)
    #                     # if eval_log_flag:
    #                     log_scale_eval(self.log, epoch, name_prefix, self.MDE[0],category, np.mean(record[0]), np.mean(record[1]), np.mean(record[2]),np.mean(record[3]),np.mean(record[4]) )

    #         if self.args['model_transfer_eval_flag']:
    #             if epoch % self.args['model_transfer_eval_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 from load_model import integrated_mde_models
    #                 unknown_MDE=copy.deepcopy(integrated_mde_models)
    #                 unknown_MDE.remove(self.args['depth_model'])
    #                 for model_name in tqdm(unknown_MDE, desc='Transfer testing'):
    #                     self.MDE = None
    #                     torch.cuda.empty_cache()
    #                     try:
    #                         self.MDE = load_target_model(model_name, self.args)
    #                     except urllib.error.URLError:
    #                         print(model_name,'cannot load during evalution')
    #                         continue
    #                     except Exception as e:
    #                         traceback.print_exc()
    #                         print('ERROR!!!!!!')
    #                         exit(-1)
    #                     for category in ['pas', 'car', 'obs']:
    #                         record = self.eval(self.MDE, category, eot_test) 
    #                         # if eval_log_flag:
    #                         log_scale_eval(self.log, epoch, name_prefix, self.MDE[0],category, np.mean(record[0]), np.mean(record[1]), np.mean(record[2]),np.mean(record[3]),np.mean(record[4]) )

    #                 self.MDE = None
    #                 torch.cuda.empty_cache()
    #                 self.MDE = load_target_model(self.args['depth_model'], self.args)
                         
    # def run_use_adam(self, scene_dir, scene_file, idx, points):
        
    #     eot_train = self.args['eot_train_flag'] 
    #     eot_test =  self.args['eot_test_flag'] 
    #     print(eot_train, eot_test)
        
    #     w = torch.atanh(self.patch.optmized_patch*2-1).detach()
    #     w.requires_grad_()
    #     opt = Adam([w],self.args['learning_rate'])
    #     self.patch.optmized_patch = 0.5*(torch.tanh(w)+1)
        
    #     self.env = ENV(self.args, scene_file, idx, points)
    #     self.env.load_scene_warp(scene_file, scene_dir)
    #     name_prefix=f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
        
    #     for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            
    #         style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
    #         if self.args['grad_type'] in ['base', 'omi']:
    #             object_insert_param, star_idx, object_num, category = self.env.accept_help(self.objects.object_imgs_train,category=None)
    #             batch,_ = self.env.accept_patch_and_objects(self.patch.optmized_patch, self.patch.init_patch, self.patch.mask, self.objects.object_imgs_train, self.env.insert_range, None,eot_train, object_insert_param, star_idx, object_num, category)
    #             batch_y = predict_batch(batch, self.MDE)
    #             adv_loss = self.adv_loss_fn(batch, batch_y)
    #             loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
    #             # grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
    #         else:
    #             raise NotImplementedError
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #         self.patch.optmized_patch = 0.5*(torch.tanh(w)+1)
        
    #         # self.patch.optmized_patch = torch.clip(self.patch.optmized_patch, 0., 1.).detach()
    #         # self.patch.optmized_patch.requires_grad_()
    #         # merged_grad += grad
    #         # if (epoch+1) % self.args['opt_step'] ==0:
    #         #     merged_grad /= self.args['opt_step']
    #         #     self.patch.optmized_patch = bim(merged_grad,self.patch.optmized_patch,self.args['learning_rate'])
    #         #     merged_grad = 0
            
    #         #log 
    #         if self.args['train_log_flag']:
    #             if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 log_img_train(self.log, epoch, name_prefix, batch, batch_y)
    #             if epoch % self.args['train_scale_log_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
    #                 log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)

    #         if self.args['inner_eval_flag']:
    #             if epoch % self.args['inner_eval_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 for category in ['pas', 'car', 'obs']:
    #                     if eot_test:
    #                         record = [[] for i in range(5)]
    #                         for _ in range(20):
    #                             record_tmp = self.eval(self.MDE, category, True)
    #                             for i in range(5):
    #                                 record[i]+=record_tmp[i]
    #                     else:
    #                         record = self.eval(self.MDE, category, False)
    #                     # if eval_log_flag:
    #                     log_scale_eval(self.log, epoch, name_prefix, self.MDE[0],category, np.mean(record[0]), np.mean(record[1]), np.mean(record[2]),np.mean(record[3]),np.mean(record[4]) )
            
    #         if self.args['model_transfer_eval_flag']:
    #             if epoch % self.args['model_transfer_eval_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 from load_model import integrated_mde_models
    #                 unknown_MDE=copy.deepcopy(integrated_mde_models)
    #                 unknown_MDE.remove(self.args['depth_model'])
    #                 for model_name in tqdm(unknown_MDE, desc='Transfer testing'):
    #                     self.MDE = None
    #                     torch.cuda.empty_cache()
    #                     try:
    #                         self.MDE = load_target_model(model_name, self.args)
    #                     except urllib.error.URLError:
    #                         print(model_name,'cannot load during evalution')
    #                         continue
    #                     except Exception as e:
    #                         traceback.print_exc()
    #                         print('ERROR!!!!!!')
    #                         exit(-1)
    #                     for category in ['pas', 'car', 'obs']:
    #                         record = self.eval(self.MDE, category, eot_test) 
    #                         # if eval_log_flag:
    #                         log_scale_eval(self.log, epoch, name_prefix, self.MDE[0],category, np.mean(record[0]), np.mean(record[1]), np.mean(record[2]),np.mean(record[3]),np.mean(record[4]) )

    #                 self.MDE = None
    #                 torch.cuda.empty_cache()
    #                 self.MDE = load_target_model(self.args['depth_model'], self.args)
      
    
    # def phy_eval_digital_load(self, scene_dir, scene_file, idx, points, insert_height, patch_height):
    #     self.env = ENV(self.args, scene_file, idx, points)
    #     self.env.load_scene_warp(scene_file, scene_dir)
    #     adv_scene_image, ben_scene_image, patch_size, patch_full_mask = self.env.accept_patch(self.patch.optmized_patch, self.patch.init_patch, self.patch.mask, insert_height, 0, patch_height = patch_height)
    #     batch= [adv_scene_image, ben_scene_image, self.env.env, patch_full_mask, self.objects.object_full_mask]
    #     batch_y = predict_batch(batch, self.MDE)
    #     log_img_train(self.log, 0, 'phy_eval', batch, batch_y)
    
    # def phy_eval_print(self, scene_dir, scene_file, idx, points):
    #     self.env = ENV(self.args, scene_file, idx, points)
    #     self.env.load_scene_warp(scene_file, scene_dir)
    #     self.objects = OBJ(self.args)
    #     self.objects.load_object_mask(self.args['phy_mask_file'], self.args['phy_mask_dir'])
        
    #     depth, _ = predict_depth_fn(self.MDE, torch.nn.functional.interpolate(self.env.env,size=([self.args['input_height'],self.args['input_width']])))
    #     print(depth.shape)
    #     mean_ori = torch.sum(depth*self.objects.object_full_mask)/torch.sum(self.objects.object_full_mask)
    #     print(mean_ori)
        
    #     colormap = plt.get_cmap('viridis')
    #     depth = depth.detach().cpu()[0][0]
    #     depth = (depth - depth.min()) / (depth.max() - depth.min())
    #     depth = colormap(depth.numpy())[..., :3]
    #     print(depth.shape)
    #     self.log.add_image(f'phy_eval_on_print', depth, 0, dataformats='HWC')
       
    # def phy_run_lbfgs_random_object(self, scene_dir, scene_file, idx, points):
    #     self.env = ENV(self.args, scene_file, idx, points)
    #     self.env.load_scene_warp(scene_file, scene_dir)
    #     patch_size_large=self.args['patch_size'].split(',')
    #     patch_size_large = [int(i) for i in patch_size_large]
    #     optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
    #     from lr_decay import PolynomialLRDecay
    #     LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

    #     for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
    #         def closure(): 
    #             # self.args['object_v_shift'] = random.randint(3,7)
    #             self.patch.optmized_patch.data.clamp_(0, 1)

    #             object_insert_param, star_idx, object_num, category = self.env.accept_help(self.objects.object_imgs_train,category=None,eot_object= False)
    #             batch, patch_size = self.env.accept_patch_and_objects(self.patch.optmized_patch, self.patch.init_patch, self.patch.mask, self.objects.object_imgs_train, self.env.insert_range, None,False, object_insert_param, star_idx, object_num, category, phy_shift_flag=self.args['phy_flag'], eot_patch=self.args['eot_train_flag'], eot_object=self.args['eot_train_flag'])
    #             # if self.args['up']==self.args['bottom']:
    #             #     try:
    #             #         assert patch_size_large[0] == patch_size[0]
    #             #         assert patch_size_large[1] == patch_size[1]
    #             #     except:
    #             #         print(f"assert error!!! given patch size:{patch_size_large}, calculated patch size:{patch_size}")
    #             #         exit()
    #             batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
    #             batch_y = predict_batch(batch, self.MDE)
    #             adv_loss = self.adv_loss_fn(batch, batch_y)
    #             mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
    #             adv_loss = self.adv_loss_fn(batch, batch_y)
    #             style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
    #             loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
    #             loss.backward()
       
    #             if (epoch+1)%10==0:
    #                 optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
    #                 self.patch.optmized_patch.data = optmized_patch.data

    #             if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 log_img_train(self.log, epoch, 'phy', batch, batch_y)
    #                 optmized_patch = (self.patch.optmized_patch*255).int().float()/255. 
    #                 self.log.add_image(f'phy/train/patch', optmized_patch.detach().cpu()[0], epoch)
    #                 log_scale_train(self.log,epoch, f'phy', style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
    #             if epoch % self.args['inner_eval_interval']==0 or (epoch+1)==self.args['epoch']: 
    #                 optmized_patch = (self.patch.optmized_patch*255).int().float()/255. 
    #                 for category in ['car']:
    #                     record = [[] for i in range(5)]
    #                     for _ in range(10):
    #                         record_tmp = self.eval(self.MDE, category, False, phy_shift_flag=self.args['phy_shift_flag'], eot_object=self.args['eot_test_flag'], eot_patch=self.args['eot_test_flag'])
    #                         for i in range(5):
    #                             record[i]+=record_tmp[i]
    #                     log_scale_eval(self.log, epoch, 'phy', self.MDE[0],category, np.mean(record[0]), np.mean(record[1]), np.mean(record[2]),np.mean(record[3]),np.mean(record[4]) )              
    #             if (epoch+1)==self.args['epoch']:
    #                 print(f'mean_ori:{mean_ori.item():.2f}\tmean_shift:{mean_shift.item():.2f}\tmrsr:{(mean_shift/mean_ori).item():.2f}') 
    #             return loss
    #         optimizer.zero_grad()
    #         optimizer.step(closure)
    #         LR_decay.step()
    #         self.patch.optmized_patch.data.clamp_(0, 1)

    # def phy_run_lbfgs_given_object(self, scene_dir, scene_file, idx, points, patch_size = None):
        
    #     self.physical_init(patch_size)

    #     self.env = ENV(self.args, scene_file, idx, points)
    #     self.env.load_scene_warp(scene_file, scene_dir)
    #     patch_size_large=self.args['patch_size'].split(',')
    #     patch_size_large = [int(i) for i in patch_size_large]
    #     patch_height = self.args['patch_height']
    #     import torch.optim as optim
    #     optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
    #     from lr_decay import PolynomialLRDecay
    #     LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

    #     for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
    #         def closure(): 
    #             # self.args['object_v_shift'] = random.randint(3,7)
    #             self.patch.optmized_patch.data.clamp_(0, 1)
    #             optimizer.zero_grad()
    #             #syn image on the scene image with original resolution
    #             #new
    #             adv_scene_image, ben_scene_image, patch_size, patch_full_mask = self.env.accept_patch(self.patch.optmized_patch, self.patch.init_patch, self.patch.mask, self.args['insert_height'], patch_height=self.args['patch_height'], phy_shift_flag=self.args['phy_flag'], eot=self.args['eot_train_flag'])
    #             batch= [adv_scene_image, ben_scene_image, self.env.env, patch_full_mask, self.objects.object_full_mask]

                
    #             if self.args['up']==self.args['bottom']:
    #                 try:
    #                     assert patch_size_large[0] == patch_size[0]
    #                     assert patch_size_large[1] == patch_size[1]
    #                 except:
    #                     print(f"assert error!!! given patch size:{patch_size_large}, calculated patch size:{patch_size}")
    #                     exit()
                
                
    #             batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch
    #                 # torch.nn.functional.interpolate(adv_scene_image,size=([int(self.args['input_height']*ra),int(self.args['input_width']*ra)])),
    #                 # torch.nn.functional.interpolate(ben_scene_image,size=([int(self.args['input_height']*ra),int(self.args['input_width']*ra)])),
    #                 # torch.nn.functional.interpolate(self.env.env,size=([int(self.args['input_height']*ra),int(self.args['input_width']*ra)])),
    #                 # torch.nn.functional.interpolate(patch_full_mask,size=([int(self.args['input_height']*ra),int(self.args['input_width']*ra)])),
    #                 # torch.nn.functional.interpolate(self.objects.object_full_mask,size=([int(self.args['input_height']*ra),int(self.args['input_width']*ra)]))
    #                 ]
    #             batch_y = predict_batch(batch, self.MDE)
    #             adv_loss = self.adv_loss_fn(batch, batch_y)
    #             mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
    #             adv_loss = self.adv_loss_fn(batch, batch_y)
    #             style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
    #             loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
    #             loss.backward()
                
    #             if (epoch+1)%10==0:
    #                 optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
    #                 self.patch.optmized_patch.data = optmized_patch.data

    #             if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
    #                 log_img_train(self.log, epoch, 'phy', batch, batch_y)
    #                 optmized_patch = (self.patch.optmized_patch*255).int().float()/255. 
    #                 self.log.add_image(f'phy/train/patch', optmized_patch.detach().cpu()[0], epoch)
    #                 self.log.add_image(f'phy/train/ben_h', ben_scene_image.detach().cpu()[0], epoch)
    #                 self.log.add_image(f'phy/train/adv_h', adv_scene_image.detach().cpu()[0], epoch)
    #                 log_scale_train(self.log,epoch, f'phy', style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)  
                              
    #             if (epoch+1)==self.args['epoch']:
    #                 print(f'mean_ori:{mean_ori.item():.2f}\tmean_shift:{mean_shift.item():.2f}\tmrsr:{(mean_shift/mean_ori).item():.2f}') 
    #             return loss
            
    #         optimizer.step(closure)
    #         LR_decay.step()
    #         self.patch.optmized_patch.data.clamp_(0, 1)

    # def phy_run(self, scene_dir, scene_file, idx, points):
        
        insert_range = [self.args['up'],self.args['bottom']]
        eot_train = self.args['eot_train_flag'] 
        self.env = ENV(self.args, scene_file, idx, points)
        self.env.load_scene_warp(scene_file, scene_dir)
        # name_prefix=f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
        merged_grad = 0
        patch_size_large=self.args['patch_size'].split(',')
        patch_size_large = [int(i) for i in patch_size_large]
        patch_height = self.args['patch_height']

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            
            style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])

            #syn image on the scene image with original resolution
            insert_height = random.randint(*insert_range)
            adv_scene_image, ben_scene_image, patch_size, patch_full_mask = self.env.accept_patch(self.patch.optmized_patch, self.patch.init_patch, self.patch.mask, insert_height, 0, patch_height = patch_height, phy_shift_flag=True, eot=True)
            
            if self.args['up']==self.args['bottom']:
                try:
                    assert patch_size_large[0] == patch_size[0]
                    assert patch_size_large[1] == patch_size[1]
                except:
                    print(f"assert error!!! given patch size:{patch_size_large}, calculated patch size:{patch_size}")
                    exit()
            batch= [
                torch.nn.functional.interpolate(adv_scene_image,size=([self.args['input_height'],self.args['input_width']])),
                torch.nn.functional.interpolate(ben_scene_image,size=([self.args['input_height'],self.args['input_width']])),
                torch.nn.functional.interpolate(self.env.env,size=([self.args['input_height'],self.args['input_width']])),
                torch.nn.functional.interpolate(patch_full_mask,size=([self.args['input_height'],self.args['input_width']])),
                torch.nn.functional.interpolate(self.objects.object_full_mask,size=([self.args['input_height'],self.args['input_width']]))
                ]
            batch_y = predict_batch(batch, self.MDE)
            mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
            print(f"{arr.item():.4f}\t{(mean_shift/mean_ori).item():.4f}", )
            adv_loss = self.adv_loss_fn(batch, batch_y)
            loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
            grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
            merged_grad += grad
            if (epoch+1) % self.args['opt_step'] ==0:
                merged_grad /= self.args['opt_step']
                self.patch.optmized_patch = bim(merged_grad,self.patch.optmized_patch,self.args['learning_rate'])
                merged_grad = 0
                
            if (epoch+1)%10==0:
                plt.imsave(f"/home/hangcheng/codes/MDE_Attack/AdvRM/phy_pics/optmized_patch/{self.args['patch_file'][:-4]}_{scene_file[:-4]}.png",self.patch.optmized_patch.detach().clone().cpu().numpy()[0].transpose([1,2,0]))
                self.patch.optmized_patch = self.patch.load_optimized_patch_only(f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}.png","/home/hangcheng/codes/MDE_Attack/AdvRM/phy_pics/optmized_patch")
                self.patch.optmized_patch.requires_grad_() 
                
            
            #log 
            if self.args['train_log_flag']:
                if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:

                    plt.imsave(f"/home/hangcheng/codes/MDE_Attack/AdvRM/phy_pics/optmized_patch/{self.args['patch_file'][:-4]}.png",self.patch.optmized_patch.detach().clone().cpu().numpy()[0].transpose([1,2,0]))
                    optmized_patch = self.patch.load_optimized_patch_only(f"{self.args['patch_file'][:-4]}.png","/home/hangcheng/codes/MDE_Attack/AdvRM/phy_pics/optmized_patch")
                    adv_scene_image, ben_scene_image, patch_size, patch_full_mask = self.env.accept_patch(optmized_patch, self.patch.init_patch, self.patch.mask, insert_height, 0, patch_height = patch_height)
                    batch= [
                            torch.nn.functional.interpolate(adv_scene_image,size=([self.args['input_height'],self.args['input_width']])),
                            torch.nn.functional.interpolate(ben_scene_image,size=([self.args['input_height'],self.args['input_width']])),
                            torch.nn.functional.interpolate(self.env.env,size=([self.args['input_height'],self.args['input_width']])),
                            torch.nn.functional.interpolate(patch_full_mask,size=([self.args['input_height'],self.args['input_width']])),
                            torch.nn.functional.interpolate(self.objects.object_full_mask,size=([self.args['input_height'],self.args['input_width']]))
                            ]
                    batch_y = predict_batch(batch, self.MDE)
                    log_img_train(self.log, epoch, 'phy_eval_print', batch, batch_y)
                    self.log.add_image(f'phy_eval_print/train/patch', optmized_patch.detach().cpu()[0], epoch)

                    mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
                    log_scale_train(self.log, epoch, 'phy_eval_print', style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)        
                
        print(f'mean_ori:{mean_ori.item():.2f}\tmean_shift:{mean_shift.item():.2f}\tmrsr:{(mean_shift/mean_ori).item():.2f}')
        # plt.imsave(f"/home/hangcheng/codes/MDE_Attack/AdvRM/phy_pics/optmized_patch/{self.args['patch_file'][:-4]}_{scene_file[:-4]}.png",self.patch.optmized_patch.detach().clone().cpu().numpy()[0].transpose([1,2,0]))
        plt.imsave(f"adv_high.png",adv_scene_image.detach().cpu().numpy()[0].transpose((1,2,0)))
        plt.imsave(f"adv_low.png", torch.nn.functional.interpolate(adv_scene_image,size=([self.args['input_height'],self.args['input_width']])).detach().cpu().numpy()[0].transpose((1,2,0)))


    def run_with_fixed_object_multi_frame(self, scene_dir, scene_file, idx, points):
        
        csv_file = self.args['csv_dir']
        scene_set =  pd.read_csv(csv_file)
        self.env1 = ENV(self.args, 'phy_ntu_1.png', scene_dir, 0, points = scene_set.iloc[0])
        self.env2 = ENV(self.args, 'phy_ntu_2.png', scene_dir, 1, points = scene_set.iloc[1])
        self.env3 = ENV(self.args, 'phy_ntu_3.png', scene_dir, 2, points = scene_set.iloc[2])
        self.env4 = ENV(self.args, 'phy_ntu_4.png', scene_dir, 3, points = scene_set.iloc[3])

        self.objects1 = OBJ(self.args)
        self.objects1.load_object_mask('phy_ntu_1_mask.png', self.args['obj_full_mask_dir'])
        self.objects2 = OBJ(self.args)
        self.objects2.load_object_mask('phy_ntu_2_mask.png', self.args['obj_full_mask_dir'])
        self.objects3 = OBJ(self.args)
        self.objects3.load_object_mask('phy_ntu_3_mask.png', self.args['obj_full_mask_dir'])
        self.objects4 = OBJ(self.args)
        self.objects4.load_object_mask('phy_ntu_4_mask.png', self.args['obj_full_mask_dir'])
        
        name_prefix1 = f"{self.args['patch_file'][:-4]}_phy_ntu_1_{idx}"
        name_prefix2 = f"{self.args['patch_file'][:-4]}_phy_ntu_2_{idx}"
        name_prefix3 = f"{self.args['patch_file'][:-4]}_phy_ntu_3_{idx}"
        name_prefix4 = f"{self.args['patch_file'][:-4]}_phy_ntu_4_{idx}"
        if self.args['update']=='lbfgs':
            optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
            LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            def closure(): 
                if self.args['update']=='lbfgs':
                    optimizer.zero_grad()
                self.patch.optmized_patch.data.clamp_(0, 1)
                
                bri,con,sat=random.uniform(0.8,1.2),random.uniform(0.9,1.1),random.uniform(0.9,1.1)
                adv_scene_image1, ben_scene_image1, patch_size1, patch_full_mask1 = self.env1.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 496, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=298,brightness=bri,contrast=con,saturation=sat)
                adv_scene_image2, ben_scene_image2, patch_size2, patch_full_mask2 = self.env2.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 446, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=219,brightness=bri,contrast=con,saturation=sat)
                adv_scene_image3, ben_scene_image3, patch_size3, patch_full_mask3 = self.env3.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 406, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=166,brightness=bri,contrast=con,saturation=sat)
                adv_scene_image4, ben_scene_image4, patch_size4, patch_full_mask4 = self.env4.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 368, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=134,brightness=bri,contrast=con,saturation=sat)

                batch1= [adv_scene_image1, ben_scene_image1, self.env1.env, patch_full_mask1, self.objects1.object_full_mask]
                batch2= [adv_scene_image2, ben_scene_image2, self.env2.env, patch_full_mask2, self.objects2.object_full_mask]
                batch3= [adv_scene_image3, ben_scene_image3, self.env3.env, patch_full_mask3, self.objects3.object_full_mask]
                batch4= [adv_scene_image4, ben_scene_image4, self.env4.env, patch_full_mask4, self.objects4.object_full_mask]
                batch1= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch1]
                batch2= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch2]
                batch3= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch3]
                batch4= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch4]

                batch_y1 = predict_batch(batch1, self.MDE)
                batch_y2 = predict_batch(batch2, self.MDE)
                batch_y3 = predict_batch(batch3, self.MDE)
                batch_y4 = predict_batch(batch4, self.MDE)
                
                adv_loss1 = self.adv_loss_fn(batch1, batch_y1)
                adv_loss2 = self.adv_loss_fn(batch2, batch_y2)
                adv_loss3 = self.adv_loss_fn(batch3, batch_y3)
                adv_loss4 = self.adv_loss_fn(batch4, batch_y4)
                adv_loss = (adv_loss1+adv_loss2+adv_loss3+adv_loss4)/4

                mean_ori1, mean_shift1, max_shift1, min_shift1, arr1 =self.eval_core(batch_y1[0], batch_y1[1], batch1[-1])
                mean_ori2, mean_shift2, max_shift2, min_shift2, arr2 =self.eval_core(batch_y2[0], batch_y2[1], batch2[-1])
                mean_ori3, mean_shift3, max_shift3, min_shift3, arr3 =self.eval_core(batch_y3[0], batch_y3[1], batch3[-1])
                mean_ori4, mean_shift4, max_shift4, min_shift4, arr4 =self.eval_core(batch_y4[0], batch_y4[1], batch4[-1])
                
                # adv_loss = self.adv_loss_fn(batch, batch_y)
                style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
                loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
                if self.args['update']=='lbfgs':
                    loss.backward()

                if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
                    optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
                    self.patch.optmized_patch.data = optmized_patch.data

                if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                    
                    log_img_train(self.log, epoch, name_prefix1, [adv_scene_image1, ben_scene_image1, (self.patch.optmized_patch*255).int().float()/255., batch_y1[0], batch_y1[1]])

                    log_img_train(self.log, epoch, name_prefix2, [adv_scene_image2, ben_scene_image2, (self.patch.optmized_patch*255).int().float()/255., batch_y2[0], batch_y2[1]])

                    log_img_train(self.log, epoch, name_prefix3, [adv_scene_image3, ben_scene_image3, (self.patch.optmized_patch*255).int().float()/255., batch_y3[0], batch_y3[1]])

                    log_img_train(self.log, epoch, name_prefix4, [adv_scene_image4, ben_scene_image4, (self.patch.optmized_patch*255).int().float()/255., batch_y4[0], batch_y4[1]])

                    log_scale_train(self.log,epoch, name_prefix1, style_score, content_score, tv_score, adv_loss, mean_shift1, max_shift1, min_shift1, mean_ori1, arr1)
                    log_scale_train(self.log,epoch, name_prefix2, style_score, content_score, tv_score, adv_loss, mean_shift2, max_shift2, min_shift2, mean_ori2, arr1)
                    log_scale_train(self.log,epoch, name_prefix3, style_score, content_score, tv_score, adv_loss, mean_shift3, max_shift3, min_shift3, mean_ori3, arr3)
                    log_scale_train(self.log,epoch, name_prefix4, style_score, content_score, tv_score, adv_loss, mean_shift4, max_shift4, min_shift4, mean_ori4, arr4)
                
                return loss
            if self.args['update']=='lbfgs':

                optimizer.step(closure)
                LR_decay.step()
            else:
                loss = closure()
                grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
                self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])