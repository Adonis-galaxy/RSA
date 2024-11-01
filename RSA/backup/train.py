import torch
import torch.backends.cudnn as cudnn
import os, sys
import argparse
import numpy as np
from dpt.models import DPTDepthModel
from utils import compute_errors, get_text, convert_arg_line_to_args
from lanscale import LanScaleModel
from loss import L1Loss, SILogLoss
from tensorboardX import SummaryWriter
from CLIP import clip
import random
import re

parser = argparse.ArgumentParser(description='LanScale', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

# LanScale
parser.add_argument("--SI_loss_lambda", type=float, default=0.85)
parser.add_argument("--scale", type=float, default=0.000305)
parser.add_argument("--shift", type=float, default=0.1378)
parser.add_argument('--dpt_model_path', type=str, default='weights/dpt_hybrid_nyu-2ce69ec7.pt')
parser.add_argument('--normalize', help='use normalize l1 loss', action='store_true')

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
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=True)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=True)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=True)
parser.add_argument('--data_path',            type=str,   help='path to the data for training', required=True)
parser.add_argument('--gt_path',              type=str,   help='path to the groundtruth data for training', required=True)
parser.add_argument('--filenames_file',       type=str,   help='path to the filenames text file for evaluation', required=True)

# Eval
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=10)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true', default=True)
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--distributed',                           help='Multiple GPU?', action='store_true', default=False)
eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=20)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=6)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--remove_lambda',         type=float, help='remove prob = lambda/box area', default=100)


# Log and save
parser.add_argument('--model_name',                type=str,   help='model name', default='lanscale')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='./models/')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq_ckpt',                 type=int,   help='Checkpoint saving frequency in global steps', default=10000)
parser.add_argument('--eval_freq',                  type=int,   help='Eval frequency in global steps', default=500)
parser.add_argument('--eval_before_train',                  action='store_true')
parser.add_argument('--load_ckpt_path',                type=str,   default=None)


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader


def eval(LanScale_model, Depth_model, CLIP_model, dataloader_eval, post_process=False, dataset=None):
    eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(dataloader_eval.data):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                print('Invalid depth. continue.')
                continue

            # Forward
            text_list = get_text(args.data_path_eval, eval_sample_batched['sample_path'], mode="eval", dataset=dataset)
            # print(eval_sample_batched['sample_path'][0].split(" ")[0],flush=True)  # Text_Ablation
            # print(text_list[0],flush=True)  # Text_Ablation

            text_tokens = clip.tokenize(text_list, truncate=True).to("cuda")
            with torch.no_grad():
                text_features = CLIP_model.encode_text(text_tokens)
            scale_pred, shift_pred = LanScale_model(text_features.float())
            # print("scale=",scale_pred[0].item(),"shift=",shift_pred[0].item(),flush=True)  # Text_Ablation

            # inverse relative depth from DPT to metric predication depth
            inv_relative_depth = Depth_model(image)
            scale_pred = scale_pred.unsqueeze(2).expand(inv_relative_depth.shape[0], inv_relative_depth.shape[1], inv_relative_depth.shape[2])
            shift_pred = shift_pred.unsqueeze(2).expand(inv_relative_depth.shape[0], inv_relative_depth.shape[1], inv_relative_depth.shape[2])
            inv_metric_depth = scale_pred * inv_relative_depth + shift_pred
            pred_depth = 1.0 / inv_metric_depth

            # Standard Eval
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

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

    if args.dataset == 'nyu':
        print("Results for sheets, NYUD2:")
        print("{:>6}, {:>6}, {:>6}, {:>6}, {:>6}, {:>6}".format('d1', 'd2', 'd3', 'abs_rel', 'log10', 'rms'))
        for i in [6, 7, 8, 1, 2, 3]:
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')

    if args.dataset == 'kitti':
        print("Results for sheets, KITTI:")
        print("{:>6}, {:>6}, {:>6}, {:>6}, {:>6}, {:>6}".format('d1', 'd2', 'd3', 'abs_rel', 'log_rms', 'rms'))
        for i in [6, 7, 8, 1, 5, 3]:
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')

    return eval_measures_cpu


def main():
    # Flag Init
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # DPT Model
    DPT_model = DPTDepthModel(
        path=args.dpt_model_path,
        scale=args.scale,
        shift=args.shift,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    DPT_model.eval()
    DPT_model.cuda()
    print("== DPT Model Initialized")

    # LanScale Model
    LanScale_model = LanScaleModel(
        text_feat_dim=1024,
        default_scale=args.scale,
        default_shift=args.shift,
        dataset=args.dataset
    )
    LanScale_model.train()
    LanScale_model.cuda()
    print("== LanScale Model Initialized")

    # CLIP Model
    CLIP_model, preprocess = clip.load("RN50", device="cuda")
    # CLIP_model, preprocess = clip.load("ViT-B/32", device="cuda")
    CLIP_model.train()
    LanScale_model.cuda()

    if args.load_ckpt_path is not None:
        checkpoint = torch.load(args.load_ckpt_path)
        LanScale_model.load_state_dict(checkpoint['model'])
        # CLIP_model.load_state_dict(checkpoint['CLIP_model'])

    # Logging
    eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
    eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    # Training Setting
    depth_loss = SILogLoss(SI_loss_lambda=args.SI_loss_lambda, max_depth=args.max_depth_eval, min_depth=args.min_depth_eval)
    # depth_loss = L1Loss(max_depth=args.max_depth_eval, min_depth=args.min_depth_eval, normalize=args.normalize)

    global_step = 1
    optimizer = torch.optim.Adam([
        {'params': LanScale_model.parameters()},
        {'params': CLIP_model.parameters()}
    ], lr=args.learning_rate)

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else args.learning_rate

    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)
    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    print("Total Steps:", num_total_steps)
    print("Save Frequency:", args.save_freq_ckpt)
    print("Start Training!")

    # Eval Before Training
    if args.eval_before_train:
        LanScale_model.eval()
        CLIP_model.eval()
        with torch.no_grad():
            eval_measures = eval(LanScale_model, DPT_model, CLIP_model, dataloader_eval, post_process=False, dataset=args.dataset)
        LanScale_model.train()
        CLIP_model.train()

    # Training Process
    init_flag = True
    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            # print("--------New Iter--------", flush=True)

            image = sample_batched['image'].cuda()  # torch.Size([B, 3, 480, 640])
            depth_gt = sample_batched['depth'].cuda()

            # Forward
            text_list = get_text(args.data_path, sample_batched['sample_path'], mode="train", remove_lambda=args.remove_lambda, dataset=args.dataset)
            text_tokens = clip.tokenize(text_list, truncate=True).to("cuda")
            with torch.no_grad():
                text_features = CLIP_model.encode_text(text_tokens)
            scale_pred, shift_pred = LanScale_model(text_features.float())

            if init_flag is True:
                init_flag = False
                print("scale:", scale_pred, flush=True)
                print("shift:", shift_pred, flush=True)
            # inverse relative depth from DPT to metric predication depth
            inv_relative_depth = DPT_model(image)
            scale_pred = scale_pred.unsqueeze(2).expand(inv_relative_depth.shape[0], inv_relative_depth.shape[1], inv_relative_depth.shape[2])
            shift_pred = shift_pred.unsqueeze(2).expand(inv_relative_depth.shape[0], inv_relative_depth.shape[1], inv_relative_depth.shape[2])
            inv_metric_depth = scale_pred * inv_relative_depth + shift_pred
            pred_depth = 1.0 / inv_metric_depth
            # BP
            loss = depth_loss(depth_prediction=pred_depth, gts=depth_gt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(LanScale_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(CLIP_model.parameters(), 1.0)
            optimizer.step()

            # Change Lr
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            # Log
            if global_step % args.log_freq == 0:
                eval_summary_writer.add_scalar("Train Loss", loss.item()/image.shape[0], int(global_step))

            # Save Checkpoitns by frequency, vis pred
            # if (global_step >= args.save_freq_ckpt and global_step % args.save_freq_ckpt ==0) or (global_step==num_total_steps):
            #     # Save CKPT
            #     model_save_name = '/model_{}'.format(global_step)
            #     print('Saving model. Step:',global_step)
            #     checkpoint = {'global_step': global_step,
            #                     'LanScale_model': LanScale_model.state_dict()}
            #     torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)

            if global_step % args.eval_freq == 0:
                LanScale_model.eval()
                CLIP_model.eval()
                with torch.no_grad():
                    print("", flush=True)
                    print("Starting Evaluation, global step=", global_step, flush=True)
                    eval_measures = eval(LanScale_model, DPT_model, CLIP_model, dataloader_eval, post_process=False, dataset=args.dataset)
                if eval_measures is not None:
                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print("")
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': LanScale_model.state_dict(),
                                          'CLIP_model': CLIP_model.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                LanScale_model.train()
                CLIP_model.train()

            global_step += 1

        epoch += 1


if __name__ == '__main__':

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)
    os.system('cp ' + "train.py" + ' ' + args_out_path + "/train.py.backup")
    os.system('cp ' + "lanscale.py" + ' ' + args_out_path + "/lanscale.py.backup")
    os.system('cp ' + "dpt/models.py" + ' ' + args_out_path + "/dpt_models.py.backup")

    main()
