# Beware of Road Markings: A Novel Adversarial Patch Against Monocular Depth Estimation
This repository contains the code of AdvRM

## Environment
```
conda env create -f environment.yaml
conda activate AdvRM
```

## Setup
**Target MDE Models.**
For models of Midas, Mono2, Mande, Dehin, DeAny, and Ada  
```
cd ..
git clone https://github.com/isl-org/MiDaS.git %Midas
git clone https://github.com/Bob-cheng/MDE_Attack.git %Mono2, Mande, Dehin
git clone https://github.com/LiheYoung/Depth-Anything.git %DeAny
git clone https://github.com/shariqfarooq123/AdaBins.git %Ada
```
DPT and GLPN are implemented in the library of transformers. 

**Scene Image.** 
In experiments, we use KITTI 3D object detection dataset as our background scene dataset. It can be downloaded [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Then you need to organize the data in the following way. The image split files can be downloaded [here](https://github.com/charlesq34/frustum-pointnets/tree/master/kitti/image_sets).
```
KITTI/object/
    
    train.txt
    val.txt
    test.txt 
    
    training/
        calib/
        image_2/ #left image
        image_3/ #right image
        label_2/
        velodyne/ 

    testing/
        calib/
        image_2/
        image_3/
        velodyne/
```
To complete realistic image synthesis, we need to annote the lane points of each selected scene image, and record them in an '.csv' file like './asset/scene/scene_lane_points.csv'. 

**Patch & Obstacle.** Put patch image 'xxx.png' or 'xxx.jpg' and its mask 'xxx_mask.png' or 'xxx_mask.jpg' in patch image folder, e.g., './asset/patch'; Put obstacle image 'xxx.png' or 'xxx.jpg' and its mask 'xxx_mask.png' or 'xxx_mask.jpg' in obstacle image folder, e.g., './asset/obstacle'; 

## Run 
1. Specify locations of MDE repositories in ./AdvRM/load_model.py
```
sys.path.append(f'{project_root}/MDE_Attack/DeepPhotoStyle_pytorch')
sys.path.append(f'{project_root}/AdaBins')
sys.path.append(f'{project_root}/MiDaS')
```

2. Demo
```
cd ./AdvRM
python run.py --depth_model monodepth2 --idx 0 --update bim  --train_log_flag --patch_file 6.jpg --scene_dir ./asset/scene --patch_dir ./asset/patch --obj_dir ./asset/obstacle --csv_dir ./asset/scene/scene_lane_points.csv --random_object_flag --train_log_flag
```

3. Check log
```
tensorboard --logdir runs
```