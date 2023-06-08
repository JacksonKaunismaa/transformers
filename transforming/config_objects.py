from dataclasses import dataclass

# design pattern: only put an entry into a Cfg class if it is the only Cfg object that manages that entry
# (ie. Cfg objects are "solely responsible" for the entries they own)

@dataclass
class ExperimentCfg:    
    # Architecture params
    vec_size: int = 1536
    n_heads: int = 12
    n_layer: int = 12
    posn_embed_type: str = "embeds"
    layer_norm_type: str = "weird"
    block_size: int = 2048
    flash: bool = True  # whether to use Flash attention or not
    dtype: str = "bfloat16"  # float32, float16, and bfloat16 are supported (for mixed precision)

    # Dropout
    dropout_attn: float = 0
    dropout_out: float = 0
    dropout_mlp: float = 0

    # Scheduler params
    lr_min: float = 1e-7 # warmup stage
    lr_max: float = 1e-3  # warmup stage
    t_warmup: int = 1500
    t_decay: int = 600_000  # should match target number of training steps

    # Training params
    total_iters: int = 600_000
    num_train: int = 6_000  # every num_train macro-batches, do an eval
    num_eval: int = 500  # do at most this many macro batches to estimate losses (train and eval) at the end of an epoch
    # epochs: int = 50
    grad_clip: float = 1.0
    weight_decay: float = 0
    batch_size: int = 32
    grad_accum_steps: int = 4  # batch_size * grad_accum_steps * block_size = num tokens per batch (262,144)
    compile: bool = False # whether or not to compile the model
    ddp: bool = False  # whether or not to use DistributedDataParallel

    def __hash__(self):
        return hash(str(self))
    

@dataclass
class DatasetCfg:    # to make things easy to pass around and access/save/compare
    dataset_path: str = ""  # based on directory
    chunk_size: int = 50  # how far apart each sample (each of size block_size) is in the dataset
    device: str = ""
    num_workers: int = 4    # number of dataloader workers

    # to be filled in when creating dataset, dont set these
    vocab_size: int = -1
    total_size: int = -1  # number of tokens in the dataset


# chunksize is how finegrained to split the dataset. sequences are sampled from the beginning of chunks,
# each of which are chunksize in size. Also defines the len of the Dataset
# self.chunk_size = chunk_size
# aaaaaaaAaaaaa|bBbbbbbbbBbbb|cccCcccccccCcc|ddddDdddddddDd|eeeeeEeeeeee|FfffffffFffff|ggGgggggggggg
# -------^-------^-------^-------^-------^-------^-------^-------^-------^-------^-------^
#       1*c     2*c     3*c     4*c     5*c     6*c     7*c     8*c     9*c     10*c    11*c
# batches are: aaaaaaaAaaaaa, Aaaaaa|bB, BbbbbbbbB, Bbbb|cccC, CcccccccC, 
#              Ccc|ddddD, DdddddddD, Dd|eeeeeE, Eeeeeee|F, FfffffffF, Fffff|ggG, Ggggggggggg

    