from argparse import ArgumentParser
from load_model import integrated_mde_models
def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--device', type= int, default=0)
    # target model 
    arg_parser.add_argument('--depth_model', type=str, default='manydepth',)
    # choices=integrated_mde_models)
    # loss weight
    arg_parser.add_argument('--style_weight', type= float, default=1000000.)
    arg_parser.add_argument('--content_weight', type= float, default=1000000.)
    arg_parser.add_argument('--tv_weight', type= float, default=0.000001)
    arg_parser.add_argument('--adv_weight', type= float, default=1000000.)
    arg_parser.add_argument('--lambda', type= float, default=0.00001)
    arg_parser.add_argument('--beta', type= float, default=1.)
    # optimization
    arg_parser.add_argument('--learning_rate', type= float, default=0.01)
    arg_parser.add_argument('--decay', type= float, default=0.1)
    arg_parser.add_argument('--epoch', type= int, default=500)
    arg_parser.add_argument('--update', type= str, default='bim',choices=['bim','lbfgs'])
    arg_parser.add_argument('--grad_type', type=str, default="base", choices=['base', 'ig', 'omi', 'igomi', 're'])
    # syn
    arg_parser.add_argument('--up', type= int, default=230) #
    arg_parser.add_argument('--bottom', type= int, default=230)
    arg_parser.add_argument('--insert_height', type= int, default=230)
    arg_parser.add_argument('--patch_height', type= int, default=70)
    arg_parser.add_argument('--ratio', type= float, default=0.75)
    arg_parser.add_argument('--h_w_ratio', type= float, default=0.75)
    arg_parser.add_argument('--object_v_shift', type= int, default=0)
    # dataset
    arg_parser.add_argument('--input_width', type= int, default=1024)
    arg_parser.add_argument('--input_height', type= int, default=320)
    arg_parser.add_argument('--sce_width', type= int, default=1024)
    arg_parser.add_argument('--sce_height', type= int, default=320)
    arg_parser.add_argument('--scene_num', type=int, default=100)
    arg_parser.add_argument('--train_car_num', type=int, default=1)
    arg_parser.add_argument('--train_pas_num', type=int, default=1)
    arg_parser.add_argument('--train_obs_num', type=int, default=1)
    # flag
    arg_parser.add_argument('--test_code', type=str, default='no')
    arg_parser.add_argument('--obj_type_train', type=str, default="car",choices=['all','pas','car','obs','npas','ncar','nobs'])
    arg_parser.add_argument('--obj_type_test', type=str, default="car",choices=['all','pas','car','obs','npas','ncar','nobs'])
    
    arg_parser.add_argument('--train_offset_object_flag', help="", action="store_true")
    arg_parser.add_argument('--test_offset_object_flag', help="", action="store_true")
    arg_parser.add_argument('--train_offset_patch_flag', help="", action="store_true")
    arg_parser.add_argument('--test_offset_patch_flag', help="", action="store_true")
    arg_parser.add_argument('--train_color_object_flag', help="", action="store_true")
    arg_parser.add_argument('--test_color_object_flag', help="", action="store_true")
    arg_parser.add_argument('--train_color_patch_flag', help="", action="store_true")
    arg_parser.add_argument('--test_color_patch_flag', help="", action="store_true")
    arg_parser.add_argument('--train_quan_patch_flag', help="", action="store_true")
    arg_parser.add_argument('--test_quan_patch_flag', help="", action="store_true")
    arg_parser.add_argument('--random_test_flag', help="", action="store_true")
    arg_parser.add_argument('--random_object_flag', help="", action="store_true")
    arg_parser.add_argument("--train_log_flag",help="", action="store_true")
    arg_parser.add_argument("--inner_eval_flag",help="", action="store_true")
    arg_parser.add_argument('--model_transfer_eval_flag', help="", action="store_true")


    # interval
    arg_parser.add_argument('--train_img_log_interval', type=int, default=100)
    arg_parser.add_argument('--train_scale_log_interval', type=int, default=100)
    arg_parser.add_argument('--inner_eval_interval', type=int, default=100)
    arg_parser.add_argument('--model_transfer_eval_interval', type=int, default=100)
    arg_parser.add_argument('--opt_step', type= int, default=1)
    # dir
    arg_parser.add_argument('--log_dir_comment', type=str, default="")
    arg_parser.add_argument('--log_dir', type=str, default="runs",help="path to save tensorboard logs")
    arg_parser.add_argument('--csv_dir', type=str, default="/home/wh/nas/MDE_Attack/AdvRM/scene_set.csv",help='path of the file recording lane points')
    arg_parser.add_argument('--obj_dir', type=str, default="/home/wh/nas/MDE_Attack/KITTI/object/object",help='path of obstacle images and their masks')
    arg_parser.add_argument('--patch_file', type=str, default="6.jpg",help='specifical patch file')
    arg_parser.add_argument('--patch_dir', type=str, default="/home/wh/nas/MDE_Attack/KITTI/object/patch", help='path of patch images and their masks')
    arg_parser.add_argument('--scene_dir', type=str, default="/home/wh/nas/MDE_Attack/KITTI/object/training/image_2",help='patch of scene images')
    arg_parser.add_argument('--obj_full_mask_dir', type=str, default="")
    arg_parser.add_argument('--obj_full_mask_file', type=str, default="")

    # random seed
    arg_parser.add_argument('--seed', type=int, default=111)
    #
    arg_parser.add_argument('--patch_size', type=str, default="")
    arg_parser.add_argument('--idx', type=int, default=-1)
    args = arg_parser.parse_args()
    args = vars(args)
    return args