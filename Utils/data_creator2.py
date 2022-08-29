from Models.StyleGan2.model import Generator
from pathlib import Path
from torchvision import utils
from tqdm import tqdm
import torch.utils.data
import numpy as np
from Configs import Global_Config
import os
BASE_PATH = Global_Config.BASE_PATH

IMAGE_DATA_DIR = BASE_PATH + 'fake/test_image/'
W_DATA_DIR = BASE_PATH + 'fake/test_w/'

network_pkl = ''
import dnnlib
import legacy
print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
    generator = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

os.makedirs(IMAGE_DATA_DIR, exist_ok=True)
os.makedirs(W_DATA_DIR, exist_ok=True)
label = torch.zeros([1, generator.c_dim], device=device)

NUMBER_OF_IMAGES = 128  # 128 / 70000  test / train

counter = 0
cur_dir = 0
num_of_images_in_single_loop = 1
latents_total = []
for i in tqdm(range(NUMBER_OF_IMAGES // num_of_images_in_single_loop)):
    with torch.no_grad():
        z = torch.randn(num_of_images_in_single_loop, 512, device='cuda')
        ws = generator.mapping(z,label,truncation=0.7)
        sample = generator.synthesis(ws, noise_mode='const')

    for index in range(len(sample)):
        if (counter % 1000) == 0:
            Path(f"{W_DATA_DIR}{int(counter / 1000)}").mkdir(parents=True, exist_ok=True)
            Path(f"{IMAGE_DATA_DIR}{int(counter / 1000)}").mkdir(parents=True, exist_ok=True)
            cur_dir = int(counter / 1000)

        with open(f'{W_DATA_DIR}{cur_dir}/{counter}.npy', 'wb') as f:
            np.save(f, ws[index][0])

        utils.save_image(
            sample[index],
            f'{IMAGE_DATA_DIR}{cur_dir}/{counter}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1)
        )
        counter += 1
