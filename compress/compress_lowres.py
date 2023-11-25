import os
import cv2
import sys
CNET_DIR = ...
sys.path.append(CNET_DIR)
import torch
import einops
import argparse
import numpy as np
from glob import glob
from PIL import Image
from tqdm import trange, tqdm
from annotator.util import HWC3
from cldm.ddim_hacked import DDIMSampler
from annotator.canny import CannyDetector
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict


DATA_DIR = ...
DATASET_NAME = 'Caltech256'
DATASET_DIR = f'{DATA_DIR}/{DATASET_NAME}'
CNET_CONFIG = f'{CNET_DIR}/models/cldm_v15.yaml'
CNET_STATE_DICT = f'{CNET_DIR}/models/control_sd15_canny.pth'
SEEDS = range(5)


def to_exemplar(path, seed=0):
    return path.replace('/train/', '/exemplars/').replace('.jpg', f'_seed{seed:02}.jpg')

def to_edge_map(path):
    return path.replace('/train/', '/edge_maps/').replace('.jpg', '.bmp')

def to_prompt(path):
    prompt = path.split('/')[-2]
    return prompt

class Compress:
    def __init__(self):
        self.apply_canny = CannyDetector()
        self.model = create_model(CNET_CONFIG).cpu()
        self.model.load_state_dict(load_state_dict(CNET_STATE_DICT, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.params = {
            'num_samples': 1,
            'a_prompt': 'clean, clear, bright, neat',
            'n_prompt': 'blurry, dark, messy, dotted',
            'strength': 1.0,
            'eta': 0.0,
            'low_threshold': 100,
            'high_threshold': 200,
        }
    
    def _resize(self, h, w):
        image_resolution = 512 if min(h, w) >= 512 else 256
        k = image_resolution / min(h, w)
        H, W = float(k * h), float(k * w)
        while H * W > 786432:
            H /= 2; W /= 2
        H, W = int(np.round(H / 64.0)) * 64, int(np.round(W / 64.0)) * 64
        return H, W

    def _save(self, array, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(array).save(path)
    
    def __call__(self, image_path, seed):
        exemplar_path = to_exemplar(image_path, seed)
        if os.path.exists(exemplar_path): return Image.open(exemplar_path)
        image = Image.open(image_path).convert('RGB')
        
        image = HWC3(np.array(image))
        h, w, _ = image.shape
        H, W = self._resize(h, w)

        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LANCZOS4 if H*W >= h*w else cv2.INTER_AREA)
        edge_map = self.apply_canny(image, self.params['low_threshold'], self.params['high_threshold'])
        self._save(edge_map, to_edge_map(image_path))
        edge_map = HWC3(edge_map)

        prompt = to_prompt(image_path)
        guess_mode = False
        ddim_steps = 50
        scale = 9.0
        
        with torch.no_grad():
            control = torch.from_numpy(edge_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(self.params['num_samples'])], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            
            seed_everything(seed)

            # self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.params['a_prompt']] * self.params['num_samples'])]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.params['n_prompt']] * self.params['num_samples'])]}
            shape = (4, H // 8, W // 8)

            # self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [self.params['strength'] * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([self.params['strength']] * 13)
            samples, _ = self.ddim_sampler.sample(ddim_steps, self.params['num_samples'], shape, cond, verbose=False, eta=self.params['eta'], unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

            # self.model.low_vram_shift(is_diffusing=False)

            x_samples = (
                einops.rearrange(self.model.decode_first_stage(samples).cpu(), 'b c h w -> b h w c') * 127.5 + 127.5
            ).numpy().clip(0, 255).astype(np.uint8)
            exemplar = cv2.resize(
                x_samples[0], (w, h),
                interpolation=(
                    cv2.INTER_LANCZOS4 if min(H, W) <= min(h, w)  # enlarge
                    else cv2.INTER_AREA  # shrink
                ),
            )
        self._save(exemplar, exemplar_path)

def get_src_paths():
    src_paths = sorted(glob(f'{DATASET_DIR}/train/*/*'))
    src_paths = [
        (path, seed)
        for seed in SEEDS
        for path in src_paths
        if not os.path.exists(to_exemplar(path, seed))
    ]
    src_paths = np.array(src_paths, dtype=object)
    return src_paths

if __name__ == '__main__':
    image_paths = get_src_paths()
    compress = Compress()
    for image_path, seed in tqdm(image_paths, desc=f'Generating'):
        compress(image_path, seed)
