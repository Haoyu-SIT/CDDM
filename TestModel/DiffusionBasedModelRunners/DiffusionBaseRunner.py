import os
from abc import ABC
from PIL import Image
from tqdm.autonotebook import tqdm
from runners.BaseRunner import BaseRunner
from runners.utils import get_image_grid


class DiffusionBaseRunner(BaseRunner, ABC):
    def __init__(self, config):
        super().__init__(config)
