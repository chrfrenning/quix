from quix.cfg import (
    RunConfig, ModelConfig, DataConfig, OptimizerConfig,
    LogConfig, add_argument
)

class GaMBiTModelConfig(ModelConfig):
    '''GaMBiT ModelConfig

    Attributes
    ----------
    n_features : int
        The number of blobs for the model.
    qkv_bias : bool
        Use bias for QKV matrices.
    '''
    n_features:int = 256
    qkv_bias:bool = True


if __name__ == '__main__':
    config = RunConfig.argparse(modcfg=GaMBiTModelConfig)
