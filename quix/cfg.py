'''Quix Config

Quix provides an extendable method to construct an config file for training 
by defining custom dataclasses. Each dataclass represents a group of related 
configuration parameters, and each field within the dataclass 
corresponds to a run parameter. 

The module is designed to be flexible and easily extendable to accommodate 
different configurations required for training PyTorch models. This is done by
simply subclassing one of the 6 config groups.

TODO: Quix config currently only supports argparse, but will soon be extended
      to verifying JSON or YAML configuration files, if this is preferrable.

Examples:
To extend the command-line arguments for custom use cases in PyTorch model training, 
you can subclass an existing configuration dataclass. Here's an example:

1. **Subclass an Existing Dataclass**:
   Quix provides a base dataclass for model configurations, `ModelConfig`, and 
   you now want to add additional arguments specific to a new model. You do this by 
   writing a subclass:

    ```python
    from quix.cfg import dataclass, ModelConfig, RunConfig, add_cfgfield

    @dataclass
    class MyModelConfig(ModelConfig):
        new_arg1: float = add_cfgfield(default=1.0, help='Description of new_arg1.')
        new_arg2: int = add_cfgfield(default=10, help='Description of new_arg2.')
    ```

2. **Usage**:
    To apply your config, you can simply do:

    ```python
    runconfig = RunConfig.argparse(modcfg=MyModelConfig)
    # Now, runconfig includes new_arg1 and new_arg2 along with base configurations
    ```

This approach of subclassing allows for quite extensive modularity in defining
and customizing the configuration for different PyTorch models or training scenarios, 
without modifying the original base configuration classes.
'''
from __future__ import annotations
import argparse
from typing import Any, Sequence, Union, Type
from dataclasses import dataclass, field, fields, is_dataclass


    
def add_cfgfield(**kwargs):
    '''Function to add arguments to dataclass fields as metadata.

    This function is used to construct arguments for dataclasses in a
    manner similar to argparse `parser.add_argument`. When extending a
    config dataclass, use this for adding custom arguments.

    In practice, this is done by populating a field's metadata with arguments
    for defaults, choices, actions, etc.
    '''
    return field(metadata = kwargs)


def _repr_helper(obj:Any, indent:int=0) -> str:
    '''Helper method for __repr__ to handle nested dataclasses.
    
    Prints in a style similar to PyTorch module printing.
    '''
    pad = ' ' * indent
    repr_str = obj.__class__.__name__ + '(\n'
    
    for fld in fields(obj):
        value = getattr(obj, fld.name)
        if is_dataclass(value):
            value_str = _repr_helper(value, indent + 2)
        else:
            value_str = repr(value)
        repr_str += f"{pad}  {fld.name}: {value_str}\n"

    repr_str += pad + ')'
    return repr_str


@dataclass
class ModelConfig:
    '''Configuration settings related to the model.
    
    Attributes:
        use_torch_zoo (bool): Flag to load model from torch models.
        use_timm_zoo (bool): Flag to load model from timm (requires timm).
        pretrained_weights (str): Path for pretrained model weights.
        sync_bn (bool): Sync Batch Norm for DDP.
    '''

    use_torch_zoo:bool = add_cfgfield(default=False, action='store_true', help='Flag to load model from torch models.')
    use_timm_zoo:bool = add_cfgfield(default=False, action='store_true', help='Flag to load model from timm (requires timm).')
    pretrained_weights:str = add_cfgfield(default=None, help='Path for pretrained model weights.')
    sync_bn:bool = add_cfgfield(default=False, action='store_true', help='Sync Batch Norm for DDP.')

    def __repr__(self):
        return _repr_helper(self, 0)


@dataclass
class AugmentationConfig:
    '''Configuration settings for data augmentation.

    Attributes:
        rrc_scale (tuple[float, float]): RandomResizeCrop scale. Default is (.08, 1.0).
        rrc_ratio (tuple[float, float]): RandomResizeCrop ratio. Default is (.75, 4/3).
        interpolation_modes (Union[str, Sequence[str]]): Interpolation modes for augmentations. 
            Default is 'all'. Accepts 'nearest', 'bilinear', 'bicubic'.
        hflip (bool): Use horizontal flip augmentation. Default is False.
        vflip (bool): Use vertical flip augmentation. Default is False.
        aug3 (bool): Use 3Augment. Default is False.
        randaug (str): RandAug level. Default is 'medium'. Choices are 'none', 'light', 'medium', 'strong'.
        cutmix_alpha (float): CutMix alpha. Default is 0.
        mixup_alpha (float): MixUp alpha. Default is 0.
    '''

    rrc_scale:tuple[float,float] = add_cfgfield(default=(.08, 1.0), help='RandomResizeCrop scale.')
    rrc_ratio:tuple[float,float] = add_cfgfield(default=(.75, 4/3), help='RandomResizeCrop ratio.')
    interpolation_modes:Union[str, Sequence[str]] = add_cfgfield(
        default='all', help='Interpolation modes for augmentations', 
        nargs='*', choices=['nearest', 'bilinear', 'bicubic']
    )
    hflip:bool = add_cfgfield(default=False, action='store_true', help='Use Aug. Hor. Flip.')
    vflip:bool = add_cfgfield(default=False, action='store_true', help='Use Aug. Ver. Flip.')
    aug3:bool = add_cfgfield(default=False, action='store_true', help='Use 3Augment.')
    randaug:str = add_cfgfield(default='medium', help='RandAug level.', type=str, choices=['none', 'light', 'medium', 'strong'])
    cutmix_alpha:float = add_cfgfield(default=0.0, help='CutMix alpha.')
    mixup_alpha:float = add_cfgfield(default=0.0, help='MixUp alpha.')

    def __repr__(self):
        return _repr_helper(self, 0)


@dataclass
class DataConfig:
    '''Configuration settings for data handling.

    Attributes:
        aug (AugmentationConfig): Augmentation configurations.
        data (str): Dataset name.
        data_path (str): Dataset path.
        workers (int): Number of workers for dataloader. Default is 16.
        prefetch (int): Prefetch factor for dataloader. Default is 2.
    '''

    aug:AugmentationConfig
    data:str = add_cfgfield(help='Dataset name.')
    data_path:str = add_cfgfield(help='Dataset path.')
    workers:int = add_cfgfield(default=16, help='Number of workers for dataloader.')
    prefetch:int = add_cfgfield(default=2, help='Prefetch factor for dataloader.')

    def __repr__(self):
        return _repr_helper(self, 0)


@dataclass
class SchedulerConfig:
    '''Configuration settings for the learning rate scheduler.

    Attributes:
        lr_scheduler (str): Learning rate scheduler type. Default is 'cosinedecay'.
        lr_warmup_epochs (int): Scheduler warmup epochs. Default is 5.
        lr_min (float): Learning rate scheduler minimum value. Default is 1e-5.
        lr_init (float): Learning rate scheduler initial value. Default is 1e-6.
    '''

    lr_scheduler:str = add_cfgfield(default='cosinedecay', help='Learning rate scheduler.')
    lr_warmup_epochs:int = add_cfgfield(default=5, help='Scheduler warmup epochs.')
    lr_min:float = add_cfgfield(default=1e-5, help='Learing rate scheduler minimum.')
    lr_init:float = add_cfgfield(default=1e-6, help='Learing rate scheduler initial value.')

    def __repr__(self):
        return _repr_helper(self, 0)


@dataclass
class OptimizerConfig:
    '''Configuration settings for the optimizer.

    Attributes:
        sch (SchedulerConfig): Scheduler configurations.
        optim (str): Name of the optimizer to use. Default is 'adamw'.
        lr (float): Learning rate (base/peak). Default is 3e-3.
        weight_decay (float): Weight decay. Default is 2e-2.
        bias_decay (float): Weight decay for bias. Default is 0.0.
        norm_decay (float): Weight decay for norm layers. Default is 0.0.
        emb_decay (float): Weight decay for transformer embeddings. Default is 0.0.
        opt_epsilon (float): Epsilon for optimizer momentum. Default is 1e-7.
        gradclip (float): Gradient norm clipping. Default is 1.0.
        accumulation_steps (int): Gradient accumulation steps. Default is 1.
        amsgrad (bool): Flag for the use of AMSGrad. Default is False.
        amp (bool): Use Automatic Mixed Precision. Default is False.
    '''

    sch:SchedulerConfig
    optim:str = add_cfgfield(default='adamw', help='Name of optimizer to use')
    lr:float = add_cfgfield(default=3e-3, help='Learning rate (base / peak).')
    weight_decay:float = add_cfgfield(default=2e-2, help='Weight decay.')
    bias_decay:float = add_cfgfield(default=0.0, help='Weight decay for bias.')
    norm_decay:float = add_cfgfield(default=0.0, help='Weight decay for norm layers.')
    emb_decay:float = add_cfgfield(default=0.0, help='Weight decay for transformer embeddings.')
    opt_epsilon:float = add_cfgfield(default=1e-7, help='Epsilon for optimizer momentum.')
    gradclip:float = add_cfgfield(default=1.0, help='Gradient norm clipping.')
    accumulation_steps:int = add_cfgfield(default=1, help='Gradient accumulation steps.')
    amsgrad:bool = add_cfgfield(default=False, action='store_true', help='Flag for use of AMSGrad.')
    amp:bool = add_cfgfield(default=False, action='store_true', help='Use Automatic Mixed Precision.')

    def __repr__(self):
        return _repr_helper(self, 0)


@dataclass
class LogConfig:
    '''Configuration settings for logging.

    Attributes:
        savedir (str): Directory for logs and model checkpoints.
        logfreq (int): Logging frequency (in iterations). Default is 50.
        stdout (bool): Flag to print logs to stdout. Default is False.
        custom_runid (str): Custom run id for logging. Default is None.
        use_neptune (bool): Flag to use Neptune logging (requires defined API key). Default is False.
    '''

    savedir:str = add_cfgfield(default='./runlogs', help='Directory for logs and model checkpoints.')
    logfreq:int = add_cfgfield(default=50, help='Logging frequency (in iterations).')
    stdout:bool = add_cfgfield(default=False, action='store_true', help='Print logs to stdout.')
    custom_runid:str = add_cfgfield(default=None, help='Custom run id for logging.')
    use_neptune:bool = add_cfgfield(default=False, action='store_true', help='Use Neptune logging (requires defined API key).')

    def __repr__(self):
        return _repr_helper(self, 0)


@dataclass
class RunConfig:
    '''Overall run configuration for the model training process.

    NOTE: This is not intended to be subclassed, however there is nothing stopping you from
          writing a custom RunConfig, however you will need to write a custom `get_arg_parser`
          function to deal with the 

    Attributes:
        mod (ModelConfig): Model configurations.
        dat (DataConfig): Data configurations.
        opt (OptimizerConfig): Optimizer configurations.
        log (LogConfig): Logging configurations.
        model (str): Model name.
        savedir (str): Directory for saving logs and model checkpoints. Default is './runlogs'.
        batch_size (int): Batch size for training. Default is 2048.
        epochs (int): Number of training epochs. Default is 300.
        test_only (bool): Flag for only performing testing / validation.
        device (str): Device type for training ('cuda' or 'cpu'). Default is 'cuda'.
        use_devices (Sequence[int]): List of device IDs to use for training. Default is [0].
        ddp_backend (str): Distributed Data Parallel (DDP) backend. Default is 'nccl'.
        ddp_init_method (str): Initialization method for DDP process group. Default is 'env://'.
        ddp_master_address (str): IP address for MASTER_ADDR (DDP). Default is 'localhost'.
        ddp_master_port (str): Port for MASTER_PORT (DDP). Default is '29500'.
    '''

    mod:ModelConfig
    dat:DataConfig
    opt:OptimizerConfig
    log:LogConfig
    model:str = add_cfgfield(help='Model name.')
    batch_size:int = add_cfgfield(default=2048, help='Number of epochs.')
    epochs:int = add_cfgfield(default=300, help='Number of epochs.')
    test_only:bool = add_cfgfield(default=False, action='store_true', help='Only perform testing / validation.')
    device:str = add_cfgfield(default='cuda', choices=['cuda', 'cpu'], help='Device type (cuda or cpu).')
    use_devices:Sequence[int] = add_cfgfield(default=[0], help='Define which devices to use.')
    ddp_backend:str = add_cfgfield(default='nccl', help='Set DDP backend for process group.')
    ddp_init_method:str = add_cfgfield(default='env://', help='Init method for DDP process group.')
    ddp_master_address:str = add_cfgfield(default='localhost', help='IP address for MASTER_ADDR (DDP).')
    ddp_master_port:str = add_cfgfield(default='29500', help='Port for MASTER_PORT (DDP).')
    
    @staticmethod
    def get_arg_parser(
        modcfg:Type[ModelConfig]=ModelConfig, 
        datcfg:Type[DataConfig]=DataConfig, 
        optcfg:Type[OptimizerConfig]=OptimizerConfig, 
        logcfg:Type[LogConfig]=LogConfig, 
        augcfg:Type[AugmentationConfig]=AugmentationConfig, 
        schcfg:Type[SchedulerConfig]=SchedulerConfig, 
    ) -> argparse.ArgumentParser:
        '''Constructs an ArgumentParser from provided configuration dataclasses.

        This function dynamically creates a command-line argument parser based on the
        fields and metadata within the provided dataclasses. Each dataclass corresponds 
        to a different configuration aspect of the model training process.

        Parameters:
            modcfg (Type[ModelConfig]): Dataclass for model configuration.
            datcfg (Type[DataConfig]): Dataclass for data handling configuration.
            optcfg (Type[OptimizerConfig]): Dataclass for optimizer configuration.
            logcfg (Type[LogConfig]): Dataclass for logging configuration.
            augcfg (Type[AugmentationConfig]): Dataclass for augmentation configuration.
            schcfg (Type[SchedulerConfig]): Dataclass for scheduler configuration.

        Returns:
            argparse.ArgumentParser: The constructed argument parser with configurations 
                                    from the provided dataclasses.
        '''

        # Main parser
        parser = argparse.ArgumentParser(description='========= QuixConfig ==========')

        # Tier 1 parsers; model, data, optimizer, logging
        modgrp = parser.add_argument_group('mod')
        datgrp = parser.add_argument_group('dat')
        optgrp = parser.add_argument_group('opt')
        loggrp = parser.add_argument_group('log')

        # Tier 2 parsers; augmentations, schedulers
        auggrp = datgrp.add_argument_group('aug')
        schgrp = optgrp.add_argument_group('sch')

        # --- Add Arguments from Config Classes ---
        grppairs = [
            (parser, RunConfig),
            (modgrp, modcfg), (datgrp, datcfg), (optgrp, optcfg), 
            (loggrp, logcfg), (auggrp, augcfg), (schgrp, schcfg)
        ]

        for arggrp, argcfg in grppairs:
            for fld in fields(argcfg):
                if fld.metadata:
                    # Fields which should be included in argparser needs metadata
                    prefix = '--' if 'default' in fld.metadata else ''
                    arggrp.add_argument(prefix + fld.name.replace('_', '-'), **fld.metadata)

        return parser
    
    @classmethod
    def from_arg_namespace(
        cls, 
        args:argparse.Namespace,
        modcfg:Type[ModelConfig]=ModelConfig, 
        datcfg:Type[DataConfig]=DataConfig, 
        optcfg:Type[OptimizerConfig]=OptimizerConfig, 
        logcfg:Type[LogConfig]=LogConfig, 
        augcfg:Type[AugmentationConfig]=AugmentationConfig, 
        schcfg:Type[SchedulerConfig]=SchedulerConfig,
    ) -> RunConfig:
        '''Factory method to create a RunConfig instance from an argument namespace.

        Parameters:
            args (argparse.Namespace): The namespace containing command-line arguments.

        Returns:
            RunConfig: An instance of RunConfig initialized with the values from args.
        '''
        # Function to extract arguments for a specific dataclass
        def extract_cfg(dc, **kwargs):
            dc_args = {}
            for fld in fields(dc):
                field_name = fld.name
                alt_field_name = field_name.replace('_', '-')
                if hasattr(args, field_name):
                    dc_args[field_name] = getattr(args, field_name)
                elif hasattr(args, alt_field_name):
                    dc_args[field_name] = getattr(args, alt_field_name)
            return dc(**dc_args, **kwargs)

        # Create tier 2 classes
        aug = extract_cfg(augcfg)
        sch = extract_cfg(schcfg)

        # Create tier 1 classes
        mod = extract_cfg(modcfg)
        dat = extract_cfg(datcfg, aug=aug)
        opt = extract_cfg(optcfg, sch=sch)
        log = extract_cfg(logcfg)

        # Combine for RunConfig
        return extract_cfg(cls, mod=mod, dat=dat, opt=opt, log=log)
    
    @classmethod
    def argparse(
        cls,
        modcfg:Type[ModelConfig]=ModelConfig, 
        datcfg:Type[DataConfig]=DataConfig, 
        optcfg:Type[OptimizerConfig]=OptimizerConfig, 
        logcfg:Type[LogConfig]=LogConfig, 
        augcfg:Type[AugmentationConfig]=AugmentationConfig, 
        schcfg:Type[SchedulerConfig]=SchedulerConfig,         
    ) -> RunConfig:
        allcfgs = (modcfg, datcfg, optcfg, logcfg, augcfg, schcfg)
        parser = cls.get_arg_parser(*allcfgs)
        args = parser.parse_args()
        return cls.from_arg_namespace(args, *allcfgs)
    

    def __repr__(self):
        return _repr_helper(self, 0)