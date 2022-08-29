import dnnlib
import legacy
import copy
import torch
import numpy as np
import Configs.Global_Config

from .torch_utils import misc
from .torch_utils.ops import conv2d_gradfix
from .torch_utils.ops import grid_sample_gradfix

conv2d_gradfix.enabled = True
grid_sample_gradfix.enabled = True

#----------------------------------------------------------------------------
#                                   有待优化
#----------------------------------------------------------------------------
# StyleGAN2 配置
cfg_specs = {
    'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
    'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
    'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
    'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
    'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
    'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
}
spec = dnnlib.EasyDict(cfg_specs['auto'])
res = 256 # 训练StyleGAN时的图像分辨率
spec.fmaps = 1 if res >= 512 else 0.5
spec.lrate = 0.002 if res >= 1024 else 0.0025
spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
spec.ema = spec.mb * 10 / 32

common_kwargs = dict(c_dim=0, img_resolution=res, img_channels=res)

G_kwargs = dnnlib.EasyDict(class_name='Models.StyleGan2_ada.model.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())

G_kwargs.synthesis_kwargs.channel_base  = int(spec.fmaps * 32768)
G_kwargs.synthesis_kwargs.channel_max  = 512
G_kwargs.mapping_kwargs.num_layers = spec.map
G_kwargs.synthesis_kwargs.num_fp16_res  = 4 # enable mixed-precision training
G_kwargs.synthesis_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow

def generator():
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).cuda() # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    
    pretrained_pkl = Configs.Global_Config.StyleGan2_ada_pretrained_pkl
    device = Configs.Global_Config.device
    with dnnlib.util.open_url(pretrained_pkl) as f:
        pretrained_data = legacy.load_network_pkl(f)
    for name,module in [('G',G),(G_ema,'G_ema')]:
        misc.copy_params_and_buffers(pretrained_data[name], module, require_all=False)

    z = torch.randn(1,512).cuda()
    c = torch.from_numpy(np.array([0])).cuda()
    t,_ = G_ema(z,c,noise_mode='const')
    
    return G_ema
