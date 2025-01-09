from typing import Literal

import torch
from ..model.encoder import EncoderCfg, get_encoder, EncoderNoPoSplat
from dacite import Config, from_dict
from omegaconf import OmegaConf
from pathlib import Path
from huggingface_hub import hf_hub_download


def load_encoder_config(cfg_path: str) -> EncoderCfg:
    return from_dict(
    EncoderCfg,
    OmegaConf.to_container(OmegaConf.load(cfg_path)),
    config=Config(type_hooks={**{Path: Path}}),
)



def init_pretrained_encoder() -> EncoderNoPoSplat:
    cfg = load_encoder_config(Path(__file__).parent / "encoder/noposplat.yaml")
    pretrained_path = download_pretrained("mixRe10kDl3dv.ckpt") #mixRe10kDl3dv_512x512.ckpt

    """
    cfg.return_depth = True
    cfg.train_depth_only = True

    if size == "large":
        cfg.num_scales = 2
        cfg.upsample_factor = 4
        cfg.lowest_feature_resolution = 8
        cfg.monodepth_vit_type="vitl"
    
        pretrained_path = download_pretrained("depthsplat-depth-large-50d3d7cf.pth")
        
    if size == "small":
        cfg.upsample_factor = 8
        cfg.lowest_feature_resolution = 8
        
        pretrained_path = download_pretrained("depthsplat-depth-small-3d79dd5e.pth")

    if size == "base":
        cfg.num_scales = 2
        cfg.upsample_factor = 4
        cfg.lowest_feature_resolution = 8
        cfg.monodepth_vit_type="vitb"

        pretrained_path = download_pretrained("depthsplat-depth-base-f57113bd.pth")


    state_dict = torch.load(pretrained_path, weights_only=True)
    """
    state_dict = torch.load(pretrained_path, weights_only=True)
    encoder, _ = get_encoder(cfg)
    
    # state keys are all prefixed with "encoder" so we need to remove that prefix
    state_dict2 = {k.replace("encoder.", ""): v for k, v in state_dict['state_dict'].items()}
    # rename 'backbone.intrinsic_weight' to 'backbone.intrinsic_encoder.weight'
    state_dict2['backbone.intrinsic_encoder.weight'] = state_dict2['backbone.intrinsic_weight']
    del state_dict2['backbone.intrinsic_weight']
    # rename 'backbone.intrinsic_bias' to 'backbone.intrinsic_encoder.bias'
    state_dict2['backbone.intrinsic_encoder.bias'] = state_dict2['backbone.intrinsic_bias']
    del state_dict2['backbone.intrinsic_bias']
    encoder.load_state_dict(state_dict2)

    return encoder

def download_pretrained(filename: str):
    return hf_hub_download(repo_id="botaoye/NoPoSplat", filename=filename, local_dir="pretrained")