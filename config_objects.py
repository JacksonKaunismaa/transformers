from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ModelCfg:    
    # Architecture params
    vec_size: int = 1536
    n_heads: int = 12
    n_layer: int = 12
    posn_embed_type: str = "embeds"
    block_size: int = 512


    attn_dropout: float = 0.1
    out_dropout: float = 0.1
    mlp_dropout: float = 0.1

    # Training params
    lr_max: float = 1e-3  # warmup stage
    t_warmup: int = 1500
    t_decay: int = 600_000
    # steps: int = 0   # so far

    grad_clip: float = 1.0
    weight_decay: float = 0
    epochs: int = 50
    batch_size: int = 128

    def __hash__(self):
        return hash(str(self))


@dataclass
class ColorDatasetConfig:
    # noise parameters
    num_noise: Tuple = (50, 70)  # number of locations to generate noise at
    noise_size: Tuple = (1, 15)  # size that each noise instance can be

    # Image parameters
    size: int = 128 # HW dimension of images
    channels: int = 1  # default is greyscale
    img_shape = (size, size, channels)  # HWC shape of input images
    bg_color: int = 0  # must have same number of channels

    # Target parameters
    task_difficulty: str = None  # function that maps colors to classes (supports iterables)
    num_classes: int = 2  # how many possible classes there are
    radius: Tuple[float, float] = (1/6., 1/3.)  # range of possible radii for circles as a fraction of size
    num_objects: int = 1 # supports ranges, if want multiclass
    overlap_attempts: int = 0  # if >0 will attempt to generate circles overlap_attempts times before giving up. if ==0 then overlaps allowed
    # Greyscale
    color_range: Tuple = (5, 255)  # range of values that the greyscale color-to-be-classified can be
    # RGB (which we generate as HSV for simplicity)
    value_range: Tuple  = (20, 100) # range for value in HSV (subset of (0, 100))
    saturation_range: Tuple = (20, 100) # range for saturation in HSV (subset of (0, 100))
    hue_range: Tuple = (0, 360)  # range for hue in HSV (subset of (0, 360))

    # Dataset Parameters
    # controlled directly by the Dataset now
    infinite = False  # if true, will not restrict itself to indices in image_indices (but will be limited lengthwise to that).
    device: str = "cpu"  # where dataloaders should get sent to
    permute_seed: int = 0  # for permuted pixels, seeds used to create the fixed permutations
    batch_size: int = 32
    num_workers: int = 6
