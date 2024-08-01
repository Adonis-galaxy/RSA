import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms
import random
import os, sys
import numpy as np
import math
import torch
from collections import Counter
import re
import inflect

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

# Do lanagugage description augmentation here
def get_text(data_path, sample_path, mode="train", dataset=None, combine_words_no_area = False):
    text_list = []
    for i in range(len(sample_path)):  # B=4
        txt_path = data_path+"/"+sample_path[i].split(' ')[0][:-4]+'.txt'
        if mode == "train":
            room_name = ""
            room_name_list = sample_path[i].split(' ')[0].split("_")[:-2]
            for i in range(len(room_name_list)):
                word = room_name_list[i]
                if i == 0:
                    room_name = room_name+word[1:]+" "
                else:
                    room_name = room_name+word+" "
        elif mode == "eval":
            room_name = sample_path[i].split(' ')[0].split("/")[0]+" "

        if dataset == "kitti":
            # room_name = "outdoor scene "
            image_area = 1216 * 352
        if dataset == "nyu":
            image_area = 480 * 640
        with open(txt_path, 'r') as file:
            # text = "A "+room_name+"with "
            text = "An image with "
            object_list = []
            area_percent_list = []
            for j, line in enumerate(file):
                # if j % 2 == 0:
                #     word = line.strip()
                #     object_list.append(word)
                # else:
                #     coords = line.split(' ')
                #     area = (float(coords[3])-float(coords[1]))*(float(coords[2])-float(coords[0]))
                #     area_list.append(area)
                object_list.append(line[:line.rfind(" ")])
                area_percent_list.append(float(line[line.rfind(" "):]))

            # remove instance based on prob=lamda/box area
            # assert len(object_list) == len(area_percent_list)
            # if mode == "train":
            #     i = 0
            #     while i < len(object_list):
            #         area_percent = area_percent_list[i]
            #         remove_prob = 1 / (1 + np.exp(-area_percent))  # sigmoid
            #         remove_prob = 1 - remove_prob
            #         # print(object_list[i], round(remove_prob,4))
            #         if random.random() < remove_prob:
            #             del object_list[i]
            #             del area_percent_list[i]
            #         else:
            #             i += 1

            # swap word as augmentation
            length = len(object_list)
            if mode == "train":
                indices = list(range(length))
                random.shuffle(indices)
                object_list = [object_list[i] for i in indices]
                area_percent_list = [area_percent_list[i] for i in indices]

            for i in range(length):
                text += object_list[i]
                # include area percent
                if combine_words_no_area is False:
                    text += " occupied " + str(round(area_percent_list[i]*100, 2)) + "% of image, "
                else:
                    text += ", "
                # text += ", "  # for combine words
                # text += ", " + str(round(area_list[i]/image_area*100, 2)) + "%; "
            if combine_words_no_area is True:
                text = combine_repetitive_words(text)  # for combine words
            text = text.replace("_", " ")

            if combine_words_no_area is True:
                text = text[:-1] + "."  # for combine words
            else:
                text = text[:-2] + "."

            # This handles nested parentheses
            pattern = r' \([^()]*\)'
            while re.search(pattern, text):
                text = re.sub(pattern, '', text)

            # print(text, flush=True)

            text_list.append(text)
    # print(text_list, flush=True)
    return text_list

def remove_repetitive_words(text):
    # Split the text into individual words
    words = text.split()

    # Keep track of encountered words
    encountered_words = set()

    # List to store unique words
    unique_words = []

    # Iterate through the words
    for word in words:
        # If the word is not encountered yet, add it to the unique_words list
        if word not in encountered_words:
            unique_words.append(word)
            encountered_words.add(word)

    # Reconstruct the string without repetitive words
    result = ' '.join(unique_words)
    return result



def combine_repetitive_words(text):
    p = inflect.engine()
    # Split the text into individual words
    words = text.split(", ")
    buffer=words[0][0:13]
    words[0]=words[0][14:]
    words.insert(0, buffer)
    words=words[:-1]
    print(words, flush=True)

    # Count the occurrences of each word
    word_counts = Counter(words)

    # Iterate through the counted words
    combined_words = []
    init=True
    for word, count in word_counts.items():
        # If the count is greater than 1, combine the word with its count
        if init is True:
            combined_words.append(word)
            if word=="An image with":
                init = False
        else:
            if count > 1:
                plural_word = p.plural(word)
                combined_words.append(f"{count} {plural_word},")
            else:
                combined_words.append(word+",")

    # Join the words back into a single string
    combined_text = ' '.join(combined_words)
    return combined_text

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])


def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused


class DistributedSamplerNoEvenlyDivisible(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        rest = len(self.dataset) - num_samples * self.num_replicas
        if self.rank < rest:
            num_samples += 1
        self.num_samples = num_samples
        self.total_size = len(dataset)
        # self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        # assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
