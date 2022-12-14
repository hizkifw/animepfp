import torch
import torch.nn as nn
from enum import Enum

import base64
import json
from io import BytesIO
from PIL import Image

# import requests
import re


class ImageType(Enum):
    REAL_UP_L = 0
    REAL_UP_R = 1
    REAL_DOWN_R = 2
    REAL_DOWN_L = 3
    FAKE = 4


def crop_image_part(image: torch.Tensor, part: ImageType) -> torch.Tensor:
    size = image.shape[2] // 2

    if part == ImageType.REAL_UP_L:
        return image[:, :, :size, :size]

    elif part == ImageType.REAL_UP_R:
        return image[:, :, :size, size:]

    elif part == ImageType.REAL_DOWN_L:
        return image[:, :, size:, :size]

    elif part == ImageType.REAL_DOWN_R:
        return image[:, :, size:, size:]

    else:
        raise ValueError("invalid part")


def init_weights(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        torch.nn.init.normal_(module.weight, 0.0, 0.02)

    if isinstance(module, nn.BatchNorm2d):
        torch.nn.init.normal_(module.weight, 1.0, 0.02)
        module.bias.data.fill_(0)


def load_image_from_local(image_path, image_resize=None):
    image = Image.open(image_path)

    if isinstance(image_resize, tuple):
        image = image.resize(image_resize)
    return image


def load_image_from_url(
    image_url, rgba_mode=False, image_resize=None, default_image=None
):
    try:
        raise NotImplementedError
        # image = Image.open(requests.get(image_url, stream=True).raw)

        if rgba_mode:
            image = image.convert("RGBA")

        if isinstance(image_resize, tuple):
            image = image.resize(image_resize)

    except Exception as e:
        image = None
        if default_image:
            image = load_image_from_local(default_image, image_resize=image_resize)

    return image


def image_to_base64(image_array):
    buffered = BytesIO()
    image_array.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64, {image_b64}"
