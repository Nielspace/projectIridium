import torch 
import torch.nn as nn
import torch.nn.functional as F

import math

import json
import copy
import logging
import os
import pathlib 
import shutil
import tempfile
import sys

from functools import wraps
from hashlib import sha256
from pathlib import Path
from io import open
from tqdm import tqdm

from urllib.parse import urlparse

#to access AWS
import boto3
import requests
from botocore.exceptions import ClientError


#---------------------------------------------------------------- url and config -----------------------------------------------------------
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'biggan-deep-128': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin",
    'biggan-deep-256': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin",
    'biggan-deep-512': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin",
}

PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'biggan-deep-128': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json",
    'biggan-deep-256': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json",
    'biggan-deep-512': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-config.json",
}


WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'config.json'

#---------------------------------------------------------------------------
#cache dir 
def designated_path_for_cache(PATH=Path.home()):
    #use PATH = Path.home() or Path(os.curdir)
    if isinstance(PATH, Path):
        PYTORCH_PRETRAINED_BIGGAN_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BIGGAN_CACHE',
                                                 PATH / '.pytorch_pretrained_biggan'))   
    else:
        raise OSError(f"'{PATH}' not found. Use Path.home() or Path(os.curdir).") 
        
    return PYTORCH_PRETRAINED_BIGGAN_CACHE




logger = logging.getLogger(__name__)



def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()

def get_from_cache(url, cache_dir=None, PATH=Path(os.curdir)):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = designated_path_for_cache(PATH)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    response = requests.head(url, allow_redirects=True)
    if response.status_code != 200:
        raise IOError("HEAD request failed for url {} with status code {}"
                        .format(url, response.status_code))
    etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'


            with open(meta_path, 'w', encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def cached_path(url_or_filename, cache_dir=None, PATH=Path(os.curdir)):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = designated_path_for_cache(PATH)
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))
        

#---------------------------------------------------------------- BigGAN ----------------------------------------------------------------







class GenBlock(nn.Module):
    def __init__(self, in_size, out_size, condition_vector_dim, reduction_factor=4, up_sample=False,
                 n_stats=51, eps=1e-12):
        super(GenBlock, self).__init__()
        self.up_sample = up_sample
        self.drop_channels = (in_size != out_size)
        middle_size = in_size // reduction_factor

        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_0 = specNorm_conv2d(in_channels=in_size, out_channels=middle_size, kernel_size=1, eps=eps)

        self.bn_1 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_1 = specNorm_conv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_2 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_2 = specNorm_conv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_3 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_3 = specNorm_conv2d(in_channels=middle_size, out_channels=out_size, kernel_size=1, eps=eps)

        self.relu = nn.ReLU()

    def forward(self, x, cond_vector, truncation):
        x0 = x

        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_1(x)

        x = self.bn_2(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_2(x)

        x = self.bn_3(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_3(x)

        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest')

        out = x + x0
        return out


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        ch = config.channel_width
        condition_vector_dim = config.z_dim * 2

        self.gen_z = specNorm_linear(in_features=condition_vector_dim,
                              out_features=4 * 4 * 16 * ch, eps=config.eps)

        layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                layers.append(SelfAttn(ch*layer[1], eps=config.eps))
            layers.append(GenBlock(ch*layer[1],
                                   ch*layer[2],
                                   condition_vector_dim,
                                   up_sample=layer[0],
                                   n_stats=config.n_stats,
                                   eps=config.eps))
        self.layers = nn.ModuleList(layers)

        self.bn = BigGANBatchNorm(ch, n_stats=config.n_stats, eps=config.eps, conditional=False)
        self.relu = nn.ReLU()
        self.conv_to_rgb = specNorm_conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=config.eps)
        self.tanh = nn.Tanh()

    def forward(self, cond_vector, truncation):
        z = self.gen_z(cond_vector[0].unsqueeze(0))

        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()

        next_available_latent_index = 1
        for layer in self.layers:
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector[next_available_latent_index].unsqueeze(0), truncation)
                next_available_latent_index += 1
            else:
                z = layer(z)

        z = self.bn(z, truncation)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, ...]
        z = self.tanh(z)
        return z



class BigGAN(nn.Module):
    """BigGAN Generator."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            model_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

        try:
            resolved_model_file = cached_path(model_file, cache_dir=cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Wrong model name, should be a valid path to a folder containing "
                         "a {} file and a {} file or a model name in {}".format(
                         WEIGHTS_NAME, CONFIG_NAME, PRETRAINED_MODEL_ARCHIVE_MAP.keys()))
            raise

        logger.info("loading model {} from cache at {}".format(pretrained_model_name_or_path, resolved_model_file))

        # Load config
        config = BigGANConfig.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        state_dict = torch.load(resolved_model_file, map_location='cpu' if not torch.cuda.is_available() else None)
        model.load_state_dict(state_dict, strict=False)
        return model

    def __init__(self, config):
        super(BigGAN, self).__init__()
        self.config = config
        self.embeddings = nn.Linear(config.num_classes, config.z_dim, bias=False)
        self.generator = Generator(config)

    def forward(self, z, class_label, truncation):
        assert 0 < truncation <= 1

        embed = self.embeddings(class_label)
        cond_vector = torch.cat((z, embed), dim=1)

        z = self.generator(cond_vector, truncation)
        return z

# resolved_config_file = cached_path("https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json")
# print(BigGANConfig.from_json_file(resolved_config_file))

