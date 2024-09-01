import torch
import torch.backends.cudnn as cudnn
import os, sys
import argparse
import numpy as np
from dpt.models import DPTDepthModel
from utils import compute_errors, get_text, convert_arg_line_to_args
from lanscale import LanScaleModel
from loss import L1Loss
from tensorboardX import SummaryWriter
from CLIP import clip
import random
import re
import matplotlib.pyplot as plt
from depth_anything.dpt import DepthAnything
from midas.model_loader import default_models, load_model
from dpt.models import DPTDepthModel
import eval_void.data_utils as data_utils
from eval_void.transforms import Transforms
from eval_void.datasets import KBNetInferenceDataset

EPS = 1e-8


def change_to_void(args):
    args.dataset = "void"
    args.txt_path_eval = "./seg_txt_void_test"
    args.do_kb_crop = False
    args.max_depth_eval = 5
    args.min_depth_eval = 0.2
    args.garg_crop = False
    args.val_image_path = "/media/home/zyzeng/code/workspace/void_1500/test_image.txt"
    args.val_sparse_depth_path = "/media/home/zyzeng/code/workspace/void_1500/test_sparse_depth.txt"
    args.val_intrinsics_path = "/media/home/zyzeng/code/workspace/void_1500/test_intrinsics.txt"
    args.val_ground_truth_path = "/media/home/zyzeng/code/workspace/void_1500/test_ground_truth.txt"


parser = argparse.ArgumentParser(description='LanScale', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

# LanScale
parser.add_argument('--loss_type', type=str, default="L1")

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Data
parser.add_argument('--distributed',                           help='Multiple GPU?', action='store_true', default=False)
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=True)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=True)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=True)
parser.add_argument('--data_path',            type=str,   help='path to the data for training', required=True)
parser.add_argument('--txt_path',            type=str,   help='path to the text for training')
parser.add_argument('--txt_path_eval',            type=str,   help='path to the text for training')
parser.add_argument('--gt_path',              type=str,   help='path to the groundtruth data for training', required=True)
parser.add_argument('--filenames_file',       type=str,   help='path to the filenames text file for evaluation', required=True)

# Eval
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=10)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true', default=True)
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=20)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=6)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--norm_loss',         action='store_true')
parser.add_argument('--combine_words_no_area',         action='store_true')

parser.add_argument('--close_car_percent',         type=float,    default=0.01)
parser.add_argument('--far_car_percent',         type=float,    default=0.001)

# Log and save
parser.add_argument('--model_name',                type=str,   help='model name', default='lanscale')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='./models/')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq_ckpt',                 type=int,   help='Checkpoint saving frequency in global steps', default=10000)
parser.add_argument('--eval_freq',                  type=int,   help='Eval frequency in global steps', default=500)
parser.add_argument('--eval_before_train',                  action='store_true')
parser.add_argument('--load_ckpt_path',                type=str,   default=None)
parser.add_argument('--two_dataset',                action='store_true')

parser.add_argument('--depth_model',                type=str, required=True)
parser.add_argument('--lambda_nyu',                type=float, default = 0.6, help="loss ratio should be around 0.5 of kitti, to 0.32 of nyu. ratio should be around 0.6")

# Eval Void
parser.add_argument('--val_image_path',
    type=str, required=False, help='Path to list of image paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, required=False, help='Path to list of sparse depth paths')
parser.add_argument('--val_intrinsics_path',
    type=str, required=False, help='Path to list of camera intrinsics paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default='', help='Path to list of ground truth depth paths')
    # Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.2, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=5.0, help='Maximum value of depth to evaluate')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

from dataloaders.dataloader import NewDataLoader

def eval(LanScale_model, depth_model, CLIP_model, dataloader_eval, ground_truths, post_process=False, dataset=None):
    eval_measures = torch.zeros(10).cuda()

    # for step, eval_sample_batched in enumerate(dataloader_eval.data):
    for idx, (image_tuple, gt_depth) in enumerate(zip(dataloader_eval, ground_truths)):
        image, image_path = image_tuple
        image = image.cuda()
        # gt_depth = gt_depth.cuda()
        image_path = [image_path[0][34:]]

        with torch.no_grad():
            # image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            # gt_depth = eval_sample_batched['depth']
            # has_valid_depth = eval_sample_batched['has_valid_depth']
            # if not has_valid_depth:
            #     print('Invalid depth. continue.')
            #     continue
            # Forward
            text_list = get_text(args.txt_path_eval, image_path, mode="eval", dataset=dataset, \
                combine_words_no_area = args.combine_words_no_area, close_car_percent=args.close_car_percent, far_car_percent = args.far_car_percent)
            text_tokens = clip.tokenize(text_list, truncate=True).to("cuda")
            text_features = CLIP_model.encode_text(text_tokens)
            scale_pred, shift_pred = LanScale_model(text_features.float())

            # For DA and Midas, do resize for image
            image_h, image_w = image.shape[2], image.shape[3]

            if args.depth_model == "da":
                a = 479 - 45
                b = 603 - 43
                image = torch.nn.functional.interpolate(
                    image,
                    size=(a, b),
                    mode="bicubic",
                    align_corners=False,
                )
            if args.depth_model == "midas":
                image = torch.nn.functional.interpolate(
                image,
                size=(384,384),
                mode="bicubic",
                align_corners=False,
            )
            relative_depth = depth_model(image) # predict relative depth
            if args.depth_model == "da" or args.depth_model == "midas":
                relative_depth = torch.nn.functional.interpolate(
                    relative_depth.unsqueeze(1),
                    size=(image_h, image_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)

            scale_pred = scale_pred.unsqueeze(2).expand(relative_depth.shape[0], relative_depth.shape[1], relative_depth.shape[2])
            shift_pred = shift_pred.unsqueeze(2).expand(relative_depth.shape[0], relative_depth.shape[1], relative_depth.shape[2])

        # linear fit
            # a = relative_depth.unsqueeze(0)
            # b = (1 / torch.from_numpy(gt_depth).cuda()).unsqueeze(0).unsqueeze(0)
            # a = torch.nn.functional.interpolate(
            #         a,
            #         size=(48, 64),
            #         mode="bilinear",
            #         align_corners=False,
            #     )
            # b = torch.nn.functional.interpolate(
            #         b,
            #         size=(48, 64),
            #         mode="bilinear",
            #         align_corners=False,
            #     )
            # a_flat = a.reshape(-1)
            # b_flat = b.reshape(-1)
            # # Construct the design matrix for linear regression
            # A = torch.stack([b_flat, torch.ones_like(b_flat)], dim=1)
            # # Solve for the least squares solution (m, n)
            # params = torch.linalg.lstsq(A,a_flat.unsqueeze(1)).solution
            # # Extract m and n
            # scale_pred = params[0]
            # shift_pred = params[1]
            # # scale_pred = torch.mean(scale_pred[~torch.isnan(scale_pred)])
            # # shift_pred = torch.mean(shift_pred[~torch.isnan(shift_pred)])
            # print(scale_pred, shift_pred)

            pred_depth = 1 / (scale_pred * relative_depth + shift_pred)

        # Median Scaling
            # relative_depth = 1 / relative_depth
            # scale_pred =  torch.from_numpy(gt_depth).median() / relative_depth.median()
            # pred_depth = scale_pred * relative_depth

            # Standard Eval
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.squeeze()
            # print(pred_depth.shape)
            # print(gt_depth.shape)
            # raise()



        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))

    if dataset == 'void':
        print("Results for sheets, VOID:")
        print("{:>6}, {:>6}, {:>6}, {:>6}, {:>6}, {:>6}".format('d1', 'd2', 'd3', 'abs_rel', 'log10', 'rms'))
        for i in [6, 7, 8, 1, 2, 3]:
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')

    return eval_measures_cpu


def main():
    # Flag Init
    # val_image_path = "/media/home/zyzeng/code/workspace/void_1500"
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    print("Depth Model:", args.depth_model)
    if args.depth_model == "da":
        # DA Model
        encoder = 'vits' # can also be 'vitb' or 'vitl'
        depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).to("cuda").eval()
        depth_model_nyu = depth_model
        depth_model_kitti = depth_model
    if args.depth_model == "dpt":
        # DPT Model
        depth_model_nyu = DPTDepthModel(
            path="weights/dpt_hybrid_nyu-2ce69ec7.pt",
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        depth_model_nyu.eval()
        depth_model_nyu.cuda()
        depth_model_kitti = DPTDepthModel(
            path="weights/dpt_hybrid_kitti-cb926ef4.pt",
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        depth_model_kitti.eval()
        depth_model_kitti.cuda()
    if args.depth_model == "midas":
        model_path = "./dpt_swin2_large_384.pt"
        device = torch.device("cuda")
        model_type = "dpt_swin2_large_384"
        depth_model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize=False, height=None, square=False)
        depth_model.eval()
        depth_model.cuda()
        depth_model_nyu = depth_model
        depth_model_kitti = depth_model

    change_to_void(args)
    print("== DPT Model Initialized")

    # LanScale Model
    LanScale_model = LanScaleModel(
        text_feat_dim=1024
    )
    LanScale_model.train()
    LanScale_model.cuda()
    print("== LanScale Model Initialized")

    # CLIP Model
    CLIP_model, preprocess = clip.load("RN50", device="cuda")
    # CLIP_model, preprocess = clip.load("ViT-B/32", device="cuda")
    CLIP_model.eval()

    # init best measures
    nyu_best_measures = torch.zeros(9).cpu()
    for i in range(6):
        nyu_best_measures[i] += 1e3
    kitti_best_measures = torch.zeros(9).cpu()
    for i in range(6):
        kitti_best_measures[i] += 1e3
    old_best_step = 0
    global_step = 1

    if args.load_ckpt_path is not None:
        checkpoint = torch.load(args.load_ckpt_path)
        LanScale_model.load_state_dict(checkpoint['model'])
        global_step = checkpoint['global_step']
        kitti_best_measures = checkpoint['best_eval_measures_kitti']
        nyu_best_measures = checkpoint['best_eval_measures_nyu']

    # Load validation data if it is available

    val_image_paths = data_utils.read_paths(args.val_image_path)
    val_sparse_depth_paths = data_utils.read_paths(args.val_sparse_depth_path)
    val_intrinsics_paths = data_utils.read_paths(args.val_intrinsics_path)
    val_ground_truth_paths = data_utils.read_paths(args.val_ground_truth_path)

    n_val_sample = len(val_image_paths)

    assert len(val_sparse_depth_paths) == n_val_sample
    assert len(val_intrinsics_paths) == n_val_sample
    assert len(val_ground_truth_paths) == n_val_sample

    ground_truths = []
    for path in val_ground_truth_paths:
        ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
        # ground_truths.append(np.stack([ground_truth, validity_map], axis=-1))
        ground_truths.append(ground_truth)

    val_dataloader = torch.utils.data.DataLoader(
        KBNetInferenceDataset(
            image_paths=val_image_paths,
            sparse_depth_paths=val_sparse_depth_paths,
            intrinsics_paths=val_intrinsics_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    val_transforms = Transforms(
        normalized_image_range=[0, 1])

    # Eval

    LanScale_model.eval()

    with torch.no_grad():
        # Void Eval
        eval_measures = eval(LanScale_model, depth_model_nyu, CLIP_model, val_dataloader, ground_truths, post_process=False, dataset="void")


if __name__ == '__main__':
    main()
    # --load_ckpt_path /media/home/zyzeng/code/workspace/LanScale/DPT_LanScale/models/0731_2d_da_panoptic_seg_txt_Huber_car_scale_tmux0/model-47500
