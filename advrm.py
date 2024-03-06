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


    # def run_with_fixed_object_multi_frame(self, scene_dir, scene_file, idx, points):
        
    #     csv_file = self.args['csv_dir']
    #     scene_set =  pd.read_csv(csv_file)
    #     self.env1 = ENV(self.args, 'phy_ntu_1.png', scene_dir, 0, points = scene_set.iloc[0])
    #     self.env2 = ENV(self.args, 'phy_ntu_2.png', scene_dir, 1, points = scene_set.iloc[1])
    #     self.env3 = ENV(self.args, 'phy_ntu_3.png', scene_dir, 2, points = scene_set.iloc[2])
    #     self.env4 = ENV(self.args, 'phy_ntu_4.png', scene_dir, 3, points = scene_set.iloc[3])

    #     self.objects1 = OBJ(self.args)
    #     self.objects1.load_object_mask('phy_ntu_1_mask.png', self.args['obj_full_mask_dir'])
    #     self.objects2 = OBJ(self.args)
    #     self.objects2.load_object_mask('phy_ntu_2_mask.png', self.args['obj_full_mask_dir'])
    #     self.objects3 = OBJ(self.args)
    #     self.objects3.load_object_mask('phy_ntu_3_mask.png', self.args['obj_full_mask_dir'])
    #     self.objects4 = OBJ(self.args)
    #     self.objects4.load_object_mask('phy_ntu_4_mask.png', self.args['obj_full_mask_dir'])
        
    #     name_prefix1 = f"{self.args['patch_file'][:-4]}_phy_ntu_1_{idx}"
    #     name_prefix2 = f"{self.args['patch_file'][:-4]}_phy_ntu_2_{idx}"
    #     name_prefix3 = f"{self.args['patch_file'][:-4]}_phy_ntu_3_{idx}"
    #     name_prefix4 = f"{self.args['patch_file'][:-4]}_phy_ntu_4_{idx}"
    #     if self.args['update']=='lbfgs':
    #         optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
    #         LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

    #     for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
    #         def closure(): 
    #             if self.args['update']=='lbfgs':
    #                 optimizer.zero_grad()
    #             self.patch.optmized_patch.data.clamp_(0, 1)
                
    #             bri,con,sat=random.uniform(0.8,1.2),random.uniform(0.9,1.1),random.uniform(0.9,1.1)
    #             adv_scene_image1, ben_scene_image1, patch_size1, patch_full_mask1 = self.env1.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 496, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=298,brightness=bri,contrast=con,saturation=sat)
    #             adv_scene_image2, ben_scene_image2, patch_size2, patch_full_mask2 = self.env2.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 446, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=219,brightness=bri,contrast=con,saturation=sat)
    #             adv_scene_image3, ben_scene_image3, patch_size3, patch_full_mask3 = self.env3.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 406, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=166,brightness=bri,contrast=con,saturation=sat)
    #             adv_scene_image4, ben_scene_image4, patch_size4, patch_full_mask4 = self.env4.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 368, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=134,brightness=bri,contrast=con,saturation=sat)

    #             batch1= [adv_scene_image1, ben_scene_image1, self.env1.env, patch_full_mask1, self.objects1.object_full_mask]
    #             batch2= [adv_scene_image2, ben_scene_image2, self.env2.env, patch_full_mask2, self.objects2.object_full_mask]
    #             batch3= [adv_scene_image3, ben_scene_image3, self.env3.env, patch_full_mask3, self.objects3.object_full_mask]
    #             batch4= [adv_scene_image4, ben_scene_image4, self.env4.env, patch_full_mask4, self.objects4.object_full_mask]
    #             batch1= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch1]
    #             batch2= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch2]
    #             batch3= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch3]
    #             batch4= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch4]

    #             batch_y1 = predict_batch(batch1, self.MDE)
    #             batch_y2 = predict_batch(batch2, self.MDE)
    #             batch_y3 = predict_batch(batch3, self.MDE)
    #             batch_y4 = predict_batch(batch4, self.MDE)
                
    #             adv_loss1 = self.adv_loss_fn(batch1, batch_y1)
    #             adv_loss2 = self.adv_loss_fn(batch2, batch_y2)
    #             adv_loss3 = self.adv_loss_fn(batch3, batch_y3)
    #             adv_loss4 = self.adv_loss_fn(batch4, batch_y4)
    #             adv_loss = (adv_loss1+adv_loss2+adv_loss3+adv_loss4)/4

    #             mean_ori1, mean_shift1, max_shift1, min_shift1, arr1 =self.eval_core(batch_y1[0], batch_y1[1], batch1[-1])
    #             mean_ori2, mean_shift2, max_shift2, min_shift2, arr2 =self.eval_core(batch_y2[0], batch_y2[1], batch2[-1])
    #             mean_ori3, mean_shift3, max_shift3, min_shift3, arr3 =self.eval_core(batch_y3[0], batch_y3[1], batch3[-1])
    #             mean_ori4, mean_shift4, max_shift4, min_shift4, arr4 =self.eval_core(batch_y4[0], batch_y4[1], batch4[-1])
                
    #             # adv_loss = self.adv_loss_fn(batch, batch_y)
    #             style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
    #             loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
    #             if self.args['update']=='lbfgs':
    #                 loss.backward()

    #             if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
    #                 optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
    #                 self.patch.optmized_patch.data = optmized_patch.data

    #             if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                    
    #                 log_img_train(self.log, epoch, name_prefix1, [adv_scene_image1, ben_scene_image1, (self.patch.optmized_patch*255).int().float()/255., batch_y1[0], batch_y1[1]])

    #                 log_img_train(self.log, epoch, name_prefix2, [adv_scene_image2, ben_scene_image2, (self.patch.optmized_patch*255).int().float()/255., batch_y2[0], batch_y2[1]])

    #                 log_img_train(self.log, epoch, name_prefix3, [adv_scene_image3, ben_scene_image3, (self.patch.optmized_patch*255).int().float()/255., batch_y3[0], batch_y3[1]])

    #                 log_img_train(self.log, epoch, name_prefix4, [adv_scene_image4, ben_scene_image4, (self.patch.optmized_patch*255).int().float()/255., batch_y4[0], batch_y4[1]])

    #                 log_scale_train(self.log,epoch, name_prefix1, style_score, content_score, tv_score, adv_loss, mean_shift1, max_shift1, min_shift1, mean_ori1, arr1)
    #                 log_scale_train(self.log,epoch, name_prefix2, style_score, content_score, tv_score, adv_loss, mean_shift2, max_shift2, min_shift2, mean_ori2, arr1)
    #                 log_scale_train(self.log,epoch, name_prefix3, style_score, content_score, tv_score, adv_loss, mean_shift3, max_shift3, min_shift3, mean_ori3, arr3)
    #                 log_scale_train(self.log,epoch, name_prefix4, style_score, content_score, tv_score, adv_loss, mean_shift4, max_shift4, min_shift4, mean_ori4, arr4)
                
    #             return loss
    #         if self.args['update']=='lbfgs':

    #             optimizer.step(closure)
    #             LR_decay.step()
    #         else:
    #             loss = closure()
    #             grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
    #             self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])