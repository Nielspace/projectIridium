import torch
import torch.nn as nn 
import torch.nn.functional as F 


import os
import sys
import subprocess
import signal
import string
import re

from datetime import datetime
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm, trange

from ema import EMA
from resample import resample
from biggan import BigGAN
from clip import load, tokenize

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


terminate = False

def signal_handling(signum,frame):
    global terminate
    terminate = True

signal.signal(signal.SIGINT,signal_handling)

# helpers

def exists(val):
    return val is not None

def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/','\\')]
    if cmd_list == None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


def create_text_path(text=None, img=None, encoding=None):
    input_name = ""
    if text is not None:
        input_name += text
    if img is not None:
        if isinstance(img, str):
            img_name = "".join(img.split(".")[:-1]) # replace spaces by underscores, remove img extension
            img_name = img_name.split("/")[-1]  # only take img name, not path
        else:
            img_name = "PIL_img"
        input_name += "_" + img_name
    if encoding is not None:
        input_name = "your_encoding"
    return input_name.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:255]

# tensor helpers

def differentiable_topk(x, k, temperature=1.):
    n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x = x.scatter(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(n, k, dim).sum(dim = 1)


def create_clip_img_transform(image_width):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
                    #T.ToPILImage(),
                    T.Resize(image_width),
                    T.CenterCrop((image_width, image_width)),
                    T.ToTensor(),
                    T.Normalize(mean=clip_mean, std=clip_std)
            ])
    return transform


def rand_cutout(image, size, center_bias=False, center_focus=2):
    width = image.shape[-1]
    min_offset = 0
    max_offset = width - size
    if center_bias:
        # sample around image center
        center = max_offset / 2
        std = center / center_focus
        offset_x = int(random.gauss(mu=center, sigma=std))
        offset_y = int(random.gauss(mu=center, sigma=std))
        # resample uniformly if over boundaries
        offset_x = random.randint(min_offset, max_offset) if (offset_x > max_offset or offset_x < min_offset) else offset_x
        offset_y = random.randint(min_offset, max_offset) if (offset_y > max_offset or offset_y < min_offset) else offset_y
    else:
        offset_x = random.randint(min_offset, max_offset)
        offset_y = random.randint(min_offset, max_offset)
    cutout = image[:, :, offset_x:offset_x + size, offset_y:offset_y + size]
    return cutout

# load clip

perceptor, normalize_image = load('ViT-B/32', jit = False)

# load biggan

class Latents(torch.nn.Module):
    def __init__(
        self,
        num_latents = 15,
        num_classes = 1000,
        z_dim = 128,
        max_classes = None,
        class_temperature = 2.
    ):
        super().__init__()
        self.normu = torch.nn.Parameter(torch.zeros(num_latents, z_dim).normal_(std = 1))
        self.cls = torch.nn.Parameter(torch.zeros(num_latents, num_classes).normal_(mean = -3.9, std = .3))
        self.register_buffer('thresh_lat', torch.tensor(1))

        assert not exists(max_classes) or max_classes > 0 and max_classes <= num_classes, f'max_classes must be between 0 and {num_classes}'
        self.max_classes = max_classes
        self.class_temperature = class_temperature

    def forward(self):
        if exists(self.max_classes):
            classes = differentiable_topk(self.cls, self.max_classes, temperature = self.class_temperature)
        else:
            classes = torch.sigmoid(self.cls)

        return self.normu, classes

class Model(nn.Module):
    def __init__(
        self,
        image_size,
        max_classes = None,
        class_temperature = 2.,
        ema_decay = 0.99
    ):
        super().__init__()
        assert image_size in (128, 256, 512), 'image size must be one of 128, 256, or 512'
        self.biggan = BigGAN.from_pretrained(f'biggan-deep-{image_size}')
        self.max_classes = max_classes
        self.class_temperature = class_temperature
        self.ema_decay\
            = ema_decay

        self.init_latents()

    def init_latents(self):
        latents = Latents(
            num_latents = len(self.biggan.config.layers) + 1,
            num_classes = self.biggan.config.num_classes,
            z_dim = self.biggan.config.z_dim,
            max_classes = self.max_classes,
            class_temperature = self.class_temperature
        )
        self.latents = EMA(latents, self.ema_decay)

    def forward(self):
        self.biggan.eval()
        out = self.biggan(*self.latents(), 1)
        return (out + 1) / 2


# class BigSleep(nn.Module):
#     def __init__(
#         self,
#         num_cutouts = 128,
#         loss_coef = 100,
#         image_size = 512,
#         bilinear = False,
#         max_classes = None,
#         class_temperature = 2.,
#         experimental_resample = False,
#         ema_decay = 0.99,
#         center_bias = False,
#     ):
#         super().__init__()
#         self.loss_coef = loss_coef
#         self.image_size = image_size
#         self.num_cutouts = num_cutouts
#         self.experimental_resample = experimental_resample
#         self.center_bias = center_bias

#         self.interpolation_settings = {'mode': 'bilinear', 'align_corners': False} if bilinear else {'mode': 'nearest'}

#         self.model = Model(
#             image_size = image_size,
#             max_classes = max_classes,
#             class_temperature = class_temperature,
#             ema_decay = ema_decay
#         )

#     def reset(self):
#         self.model.init_latents()

#     def sim_txt_to_img(self, text_embed, img_embed, text_type="max"):
#         sign = -1
#         if text_type == "min":
#             sign = 1
#         return sign * self.loss_coef * torch.cosine_similarity(text_embed, img_embed, dim = -1).mean()

#     def forward(self, text_embeds, text_min_embeds=[], return_loss = True):
#         width, num_cutouts = self.image_size, self.num_cutouts

#         out = self.model()

#         if not return_loss:
#             return out

#         pieces = []
#         for ch in range(num_cutouts):
#             # sample cutout size
#             size = int(width * torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
#             # get cutout
#             apper = rand_cutout(out, size, center_bias=self.center_bias)
#             if (self.experimental_resample):
#                 apper = resample(apper, (224, 224))
#             else:
#                 apper = F.interpolate(apper, (224, 224), **self.interpolation_settings)
#             pieces.append(apper)

#         into = torch.cat(pieces)
#         into = normalize_image(into)

#         image_embed = perceptor.encode_image(into)

#         latents, soft_one_hot_classes = self.model.latents()
#         num_latents = latents.shape[0]
#         latent_thres = self.model.latents.model.thresh_lat

#         lat_loss =  torch.abs(1 - torch.std(latents, dim=1)).mean() + \
#                     torch.abs(torch.mean(latents, dim = 1)).mean() + \
#                     4 * torch.max(torch.square(latents).mean(), latent_thres)


#         for array in latents:
#             mean = torch.mean(array)
#             diffs = array - mean
#             var = torch.mean(torch.pow(diffs, 2.0))
#             std = torch.pow(var, 0.5)
#             zscores = diffs / std
#             skews = torch.mean(torch.pow(zscores, 3.0))
#             kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0

#             lat_loss = lat_loss + torch.abs(kurtoses) / num_latents + torch.abs(skews) / num_latents

#         cls_loss = ((50 * torch.topk(soft_one_hot_classes, largest = False, dim = 1, k = 999)[0]) ** 2).mean()

#         results = []
#         for txt_embed in text_embeds:
#             results.append(self.sim_txt_to_img(txt_embed, image_embed))
#         for txt_min_embed in text_min_embeds:
#             results.append(self.sim_txt_to_img(txt_min_embed, image_embed, "min"))
#         sim_loss = sum(results).mean()
#         return out, (lat_loss, cls_loss, sim_loss)


print('done')