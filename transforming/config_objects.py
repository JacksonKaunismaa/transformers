import dataclasses

# design pattern: only put an entry into a Cfg class if it is the only Cfg object that manages that entry
# (ie. Cfg objects are "solely responsible" for the entries they own)

@dataclasses.dataclass
class ExperimentCfg:    
    # Architecture params
    vec_size: int = 1536
    n_heads: int = 12
    n_layer: int = 12
    posn_embed_learnable: bool = True
    posn_embed_type: str = "base"  # must be in ['base', 'relative']
    layer_norm_posn: str = "weird"
    block_size: int = 2048
    flash: bool = True  # whether to use Flash attention or not
    dtype: str = "bfloat16"  # float32, float16, and bfloat16 are supported (for mixed precision)
    linear_bias: bool = False  # enable/disable biases for all MLP Linear layers
    learnable_unembed: bool = True  # enable/disable having the unembed matrix be seperately learnable from the embedding matrix
    # Normalizer params
    normalizer_bias: bool = False  # enable/disable biases on the normalizer
    normalizer_eps: float = 1e-8   # eps for making sure divide by zero doesn't happen when normalizing
    normalizer_type: str = "LayerNorm"  # must be one of "LayerNorm", "RMSNorm"
    rmsnorm_p: float = 1.0  # only has effect if normalizer_type == "RMSNorm", in which case it becomes pRMSNorm


    # Dropout
    dropout_attn: float = 0
    dropout_out: float = 0
    dropout_mlp: float = 0

    # Scheduler params
    lr_min: float = 1e-7 # warmup stage
    lr_max: float = 1e-3  # warmup stage
    t_warmup: int = 1500
    t_decay: int = 40_000  # should match target number of training steps (total_steps)

    # Training params
    total_steps: int = 40_000  # set so that we see each token in the dataset 10x?
    train_steps: int = 500  # every num_train macro-batches, do an eval (yes, i realize this definition conflicts with num_eval and is confusing)
    num_eval: int = 500  # do at most this many micro batches to estimate losses (train and eval) at the end of an epoch
    # epochs: int = 50
    grad_clip: float = 1.0
    weight_decay: float = 0
    batch_size: int = 32
    grad_accum_steps: int = 4  # batch_size * num_grad_accum * block_size = num tokens per macro-batch (should be ~250_000)
    compile: bool = False # whether or not to compile the model
    ddp: bool = False   # whether or not to use DistributedDataParallel
    zero: bool = False  # whether or not to use ZeroRedundancyOptimizer (only if ddp set)
    checkpointing: bool = False   # whether or not to use activation checkpointing
    label_smoothing: float = 1e-8  # amount of label smoothing to use

    # Misc.
    job_id: int = 0  # slurm job id / true location of checkpoint

    def __hash__(self):
        return hash(str(self))
    
    def get_dry(self):
        # some defaults that are meant for running a very small network while debugging
        return dataclasses.replace(self, vec_size=128, n_layer=1, n_heads=4, lr_max=2e-4, lr_min=1e-7, block_size=1024, batch_size=2,
                grad_accum_steps=16, train_steps=2, num_eval=3, dtype="float16", compile=False, zero=False, checkpointing=False,
                normalizer_type="RMSNorm", rmsnorm_p=0.2, posn_embed_type="relative", flash=False, posn_embed_learnable=False)
        # return dataclasses.replace(self, 
        #         grad_accum_steps=8, train_steps=2, num_eval=3, dtype="float16", compile=False, zero=False, checkpointing=False,
        #         normalizer_type="RMSNorm", rmsnorm_p=0.2, posn_embed_type="relative", flash=False, posn_embed_learnable=False)
    

@dataclasses.dataclass
class DatasetCfg:    # to make things easy to pass around and access/save/compare
    dataset_path: str = ""  # based on directory
    chunk_size: int = 50  # how far apart each sample (each of size block_size) is in the dataset
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

    