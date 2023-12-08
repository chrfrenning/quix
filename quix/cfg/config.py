from __future__ import annotations
import os
import json
import yaml
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field, fields, _MISSING_TYPE
from typing import (
    Type, TypeVar, Generic, Any, Dict, Optional, ClassVar, Union,
    Sequence, List, Tuple, get_type_hints, get_args, get_origin
)
from .cfgutils import (
    metadata_decorator, _repr_helper, _get_parser, _fromenv
)

def add_argument(**kwargs):
    '''Quix parsing method to add arguments.

    Provides an API similar to argparse.ArgumentParser's add_argument.
    NOTE: `dest` arguments are currently incompatible with Quix parser.
    '''
    if 'default' in kwargs:
        default_value = kwargs['default']
        default_type = type(default_value)
        if type(default_value) in [list, dict, set]:
            return field(default_factory=default_type, metadata=kwargs)
        return field(default=default_value, metadata=kwargs)
    return field(metadata=kwargs)


class _MetaConfig(type):

    __dataclass_fields__: ClassVar[Dict] 
    
    def __new__(cls, name, bases, class_dict):
        datacls = dataclass(super().__new__(cls, name, bases, class_dict), repr=False) # type: ignore
        metacls = metadata_decorator(datacls)
        return metacls
    

class _BaseConfig(metaclass=_MetaConfig):

    def __repr__(self):
        return _repr_helper(self)


class ModelConfig(_BaseConfig):
    '''Model configuration dataclass.

    Attributes
    ----------
    model : str
        Name of model to be parsed.
    use_torch_zoo : bool
        Flag to load model from torch models.
    pretrained_weights : Optional[str]
        Pretrained model weights, passed to torchvision zoo.
    resume : Optional[str]
        Path of weights to resume training from.
    sync_bn : bool
        Sync Batch Norm for DDP.
    '''
    model: str
    use_torch_zoo: bool = True
    pretrained_weights: Optional[str] = None
    resume: Optional[str] = None
    sync_bn: bool = False


class DataConfig(_BaseConfig):
    '''Configuration settings for data handling.

    Attributes
    ----------
    data_path : str
        Dataset path.
    dataset : str
        Dataset name.
    workers : int
        Number of workers for dataloader.
    prefetch : int
        Prefetch factor for dataloader.
    input_ext : str
        Extensions for inputs in dataset.
    target_ext : str
        Extensions for targets in dataset.
    num_classes : int
        Number of classes.
    img_size : int
        Image size used for training.
    val_size : Optional[int]
        Set explicit image size for validation.
    rrc_scale : tuple[float, float] 
        RandomResizeCrop scale.
    rrc_ratio : tuple[float, float] 
        RandomResizeCrop ratio.
    intp_modes : List[str]
        Interpolation modes for augmentations.
    hflip : bool
        Use horizontal flip augmentation.
    vflip : bool
        Use vertical flip augmentation.
    jitter : bool
        Use color jitter.
    aug3 : bool
        Use 3Augment.
    randaug : str
        RandAug level.
    cutmix_alpha : float
        CutMix alpha.
    mixup_alpha : float
        MixUp alpha.
    ra_sampler : bool
        Use repeated augmentation sampler.
    ra_reps : int
        Number of repeated augmentations for RASampler.        
    '''
    data_path:str
    dataset:str = 'IN1k'
    workers:int = 16
    prefetch:int = 2
    input_ext:List[str] = add_argument(default=None, nargs='+')
    target_ext:List[str] = add_argument(default=None, nargs='+')
    num_classes:Optional[int] = None
    img_size:int = 224
    val_size:Optional[int] = None
    rrc_scale:Tuple[float,float] = add_argument(default=(.08, 1.0), nargs=2)
    rrc_ratio:Tuple[float,float] = add_argument(default=(.75, 4/3), nargs=2)
    intp_modes:str = add_argument(
        default=['nearest', 'bilinear', 'bicubic'], nargs='*', choices=['nearest', 'bilinear', 'bicubic']
    )
    hflip:bool = True
    vflip:bool = add_argument(default=False, action='store_true')
    jitter:bool = True
    aug3:bool = add_argument(default=False, action='store_true')
    randaug:str = add_argument(default='medium', choices=['none', 'light', 'medium', 'strong'])
    cutmix_alpha:float = 0.0
    mixup_alpha:float = 0.0
    ra_sampler:bool = add_argument(default=False, action='store_true')
    ra_reps:int = 2


class OptimizerConfig(_BaseConfig):
    '''Configuration settings for the optimizer.

    Attributes
    ----------
    optim : str
        Name of the optimizer to use.
    lr : float
        Learning rate [base/peak].
    weight_decay : float
        Weight decay.
    norm_decay : float
        Weight decay for norm layers.
    custom_decay_keys : str
        Keys for setting custom weight decay for certain parameters.
    custom_decay_vals : float
        Values for custom weight decay keys.
    opt_epsilon : float
        Epsilon for optimizer momentum.
    gradclip : float
        Gradient norm clipping.
    accumulation_steps : int
        Gradient accumulation steps.
    amsgrad : bool
        Flag for the use of AMSGrad.
    model_ema : bool 
        Use model exponential moving average [EMA]
    model_ema_steps : int
        Number of steps for model EMA
    model_ema_decay : float
        Decay for model EMA
    amp : bool
        Use Automatic Mixed Precision.
    consistent_batch_size : bool
        Use consistent batch size in train/val.
    smoothing : float
        Label smoothing, mainly for classification.
    lr_scheduler : str
        Learning rate scheduler type.
    lr_warmup_epochs : int
        Scheduler warmup epochs.
    lr_min : float
        Learning rate scheduler minimum value.
    lr_init : float
        Learning rate scheduler initial value.        
    '''
    optim:str = 'adamw'
    lr:float = 3e-3
    weight_decay:float = 2e-2
    norm_decay:float = 0.0
    custom_decay_keys:List[str] = add_argument(default=[], nargs='+')
    custom_decay_vals:List[float] = add_argument(default=[], nargs='+')
    opt_epsilon:float = 1e-7
    gradclip:float = 1.0
    accumulation_steps:int = 1
    amsgrad:bool = add_argument(default=False, action='store_true')
    model_ema:bool = add_argument(default=False, action='store_true')
    model_ema_steps:int = 32
    model_ema_decay:float = 0.9998
    amp:bool = add_argument(default=False, action='store_true')
    consistent_batch_size:bool = True
    smoothing:float = 0.0
    lr_scheduler:str = 'cosinedecay'
    lr_warmup_epochs:int = 5
    lr_min:float = 1e-5
    lr_init:float = 1e-6


class LogConfig(_BaseConfig):
    '''Configuration settings for logging.

    Attributes
    ----------
    savedir : str
        Directory for logs and model checkpoints.
    logfreq : int
        Logging frequency [in iterations].
    stdout : bool
        Flag to print logs to stdout.
    custom_runid : Optional[str]
        Custom run id for logging.
    '''
    savedir:Optional[str] = None
    logfreq:int = 50
    stdout:bool = add_argument(default=False, action='store_true')
    custom_runid:Optional[str] = _fromenv('RUNID', str)


TMod = TypeVar('TMod', bound=ModelConfig)
TOpt = TypeVar('TOpt', bound=OptimizerConfig)
TDat = TypeVar('TDat', bound=DataConfig)
TLog = TypeVar('TLog', bound=LogConfig)


@metadata_decorator
@dataclass(repr=False)
class RunConfig(Generic[TMod,TDat,TOpt,TLog]):
    '''Quix RunConfig.

    Attributes
    ----------
    mod : ModelConfig
        The model configuration parameters.
    dat : DataConfig
        Data configuration parameters.
    opt : OptConfig
        Optimization configuration parameters.
    log : LogConfig
        Logging configuration parameters.
    cfg : str
        Path to config file in JSON/YAML format.
    batch_size : int
        Batch size for training. 
    epochs : int
        Number of training epochs. 
    start_epoch : int
        Start training from given epoch
    test_only : bool
        Flag for only performing testing / validation.
    device : str
        Device type for training ['cuda' or 'cpu']. 
    ddp_backend : str
        Distributed Data Parallel [DDP] backend. 
    ddp_url : str
        URL for DDP process group. 
    ddp_master_address : str
        IP address for MASTER_ADDR [DDP]. 
    ddp_master_port : str
        Port for MASTER_PORT [DDP]. 
    world_size : int
        World size for DDP. If None, this is inferred from environment variables.
    rank : int
        Rank for DDP. If None, this is inferred from environment variables.
    local_rank : int
        Local device rank for DDP. If None, this is inferred from environment variables.
    '''
    mod:TMod = add_argument(no_parse=True)
    dat:TDat = add_argument(no_parse=True)
    opt:TOpt = add_argument(no_parse=True)
    log:TLog = add_argument(no_parse=True)
    cfg:str = add_argument(default=[], nargs='*')
    batch_size:int = 256
    epochs:int = 300
    start_epoch: int = 0
    test_only:bool = add_argument(default=False, action='store_true')
    ddp_backend:str = 'nccl'
    ddp_url:str = 'env://'
    ddp_master_address:str = 'localhost'
    ddp_master_port:str = '29500'
    device:str = add_argument(default='cuda', choices=['cuda', 'cpu'])
    world_size:Optional[int] = _fromenv('WORLD_SIZE', int)
    rank:Optional[int] = _fromenv('RANK', int)
    local_rank:Optional[int] = _fromenv('LOCAL_RANK', int)

    def __repr__(self):
        return _repr_helper(self)
    
    @classmethod
    def _get_required(
        cls, 
        modcfg:Type[TMod]=ModelConfig,          #type:ignore
        datcfg:Type[TDat]=DataConfig,           #type:ignore
        optcfg:Type[TOpt]=OptimizerConfig,      #type:ignore
        logcfg:Type[TLog]=LogConfig,            #type:ignore
    ) -> Sequence[str]:
        # Check if required arguments are provided
        allcfgs = [modcfg, datcfg, optcfg, logcfg]
        required = [
            f.name for cfg in [cls] + allcfgs
            for f in fields(cfg) # type: ignore
            if isinstance(f.metadata.get('default'), _MISSING_TYPE)
            and not f.metadata.get('no_parse', False)
        ]
        return required

    @staticmethod
    def get_arg_parser(
        modcfg:Type[TMod]=ModelConfig,          # type: ignore
        datcfg:Type[TDat]=DataConfig,           # type: ignore
        optcfg:Type[TOpt]=OptimizerConfig,      # type: ignore
        logcfg:Type[TLog]=LogConfig,            # type: ignore
        base_types:Sequence[Type]=[int, float, complex, str, bool]
    ) -> ArgumentParser:
        '''Constructs an ArgumentParser from provided configuration dataclasses.

        This function dynamically creates a command-line argument parser based on the
        fields and metadata within the provided dataclasses. Each dataclass corresponds 
        to a different configuration aspect of the model training process.

        Parameters
        ----------
        modcfg : Type[ModelConfig]
            Dataclass for model configuration.
        datcfg : Type[DataConfig]
            Dataclass for data handling configuration.
        optcfg : Type[OptimizerConfig]
            Dataclass for optimizer configuration.
        logcfg : Type[LogConfig]
            Dataclass for logging configuration.

        Returns
        -------
            ArgumentParser: The constructed argument parser with configurations 
                            from the provided dataclasses.
        '''

        # Main parser
        parser = ArgumentParser(description='========= Quix Base Config ==========')
        modgrp = parser.add_argument_group('mod')
        datgrp = parser.add_argument_group('dat')
        optgrp = parser.add_argument_group('opt')
        loggrp = parser.add_argument_group('log')
            
        # --- Add Arguments from Config Classes ---
        grppairs = [
            (parser, RunConfig),
            (modgrp, modcfg), (datgrp, datcfg), (optgrp, optcfg), (loggrp, logcfg)
        ]
        
        def _kwargs(name, type_hint, **metadata):
            if 'action' in metadata:
                return metadata
            origin = get_origin(type_hint)
            args = get_args(type_hint)
            descr = f'{name} ({type_hint})'
            types = args if origin in [Union, List, list, Tuple, tuple, Optional] else [type_hint]
            type_fn = _get_parser(types, base_types, descr)
            nargs = metadata.pop('nargs', {List:'*', Tuple:len(args), Optional:'?'}.get(origin, None))
            return {'type': type_fn, 'nargs':nargs, **metadata}

        for arggrp, argcfg in grppairs:
            type_hints = get_type_hints(argcfg)
            for fld in fields(argcfg):
                if fld.metadata:
                    name = '--' + fld.name.replace('_', '-')
                    if fld.metadata.get('no_parse', False):
                        pass
                    else:
                        arggrp.add_argument(name, **_kwargs(fld.name, type_hints[fld.name], **fld.metadata))

        return parser    
    
    @classmethod
    def from_namespace(
        cls,
        args:Namespace,
        modcfg:Type[TMod]=ModelConfig,          #type:ignore
        datcfg:Type[TDat]=DataConfig,           #type:ignore
        optcfg:Type[TOpt]=OptimizerConfig,      #type:ignore
        logcfg:Type[TLog]=LogConfig,            #type:ignore
    ) -> RunConfig[TMod,TDat,TOpt,TLog]:
        '''Factory method to create a RunConfig instance from an argument namespace.

        Parameters
        ----------
        args : Namespace)
            The namespace containing command-line arguments.
        modcfg : Type[ModelConfig]
            Dataclass for model configuration.
        datcfg : Type[DataConfig]
            Dataclass for data handling configuration.
        optcfg : Type[OptimizerConfig]
            Dataclass for optimizer configuration.
        logcfg : Type[LogConfig]
            Dataclass for logging configuration.

        Returns
        -------
        RunConfig
            An instance of RunConfig initialized with the values from args.
        '''
        def extract_cfg(dc, **kwargs):
            '''Extracts arguments for a specific dataclass
            '''
            dc_args = {}
            for fld in fields(dc):
                field_name = fld.name
                alt_field_name = field_name.replace('_', '-')
                if hasattr(args, field_name):
                    dc_args[field_name] = getattr(args, field_name)
                elif hasattr(args, alt_field_name):
                    dc_args[field_name] = getattr(args, alt_field_name)
            return dc(**dc_args, **kwargs)

        mod = extract_cfg(modcfg)
        dat = extract_cfg(datcfg)
        opt = extract_cfg(optcfg)
        log = extract_cfg(logcfg)

        # Combine for RunConfig
        run = extract_cfg(cls, mod=mod, dat=dat, opt=opt, log=log)

        return run
    
    @staticmethod
    def load_config(file_paths:Optional[Sequence[str]]) -> Dict[str, Any]:
        '''Loads configuration from file(s)

        Parameter
        ---------
        file_paths : Optional[Sequence[str]]
            List of configuration files, parsed in sequence. This means that each
            subsequent file can potentially override the previous one.

        Returns
        -------
        Dict[str]
            Parsed config parameters from passed files.
        '''
        config = {}
        if file_paths is not None:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Config file not found: {file_path}")                
                with open(file_path, 'r') as file:
                    if file_path.endswith('.json'):
                        config.update(json.load(file))
                    elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                        config.update(yaml.safe_load(file))
                    else:
                        raise ValueError("Unsupported config file format")

        # Fix dashes for python naming
        config = {k.replace('-', '_'):v for k,v in config.items()}
        return config
    
    @classmethod
    def argparse(
        cls,
        modcfg:Type[TMod]=ModelConfig,          #type: ignore
        datcfg:Type[TDat]=DataConfig,           #type: ignore
        optcfg:Type[TOpt]=OptimizerConfig,      #type: ignore
        logcfg:Type[TLog]=LogConfig,            #type: ignore
        **kwargs
    ) -> RunConfig[TMod,TDat,TOpt,TLog]:
        '''Parses a Quix RunConfig from command line arguments.

        Parameters
        ----------
        modcfg : Type[ModelConfig]
            Dataclass for model configuration.
        datcfg : Type[DataConfig]
            Dataclass for data handling configuration.
        optcfg : Type[OptimizerConfig]
            Dataclass for optimizer configuration.
        logcfg : Type[LogConfig]
            Dataclass for logging configuration.
        '''
        allcfgs = [modcfg, datcfg, optcfg, logcfg]
        parser = cls.get_arg_parser(*allcfgs)
        
        if '_testargs' in kwargs:
            args = parser.parse_args(kwargs.get('_testargs', []))
        else:
            args = parser.parse_args()
        
        config = cls.load_config(args.cfg)
        parser.set_defaults(**config)
        
        if '_testargs' in kwargs:
            args = parser.parse_args(kwargs.get('_testargs', []))
        else:
            args = parser.parse_args()
        
        # Check missing args
        def missing(arg):
            cur_arg = getattr(args, arg)
            return cur_arg is None or isinstance(cur_arg, _MISSING_TYPE)

        missing_args = [arg for arg in cls._get_required(*allcfgs) if missing(arg)]
        if missing_args:
            parser.error(f"Missing required arguments: {', '.join(missing_args)}")

        return cls.from_namespace(args, *allcfgs)
