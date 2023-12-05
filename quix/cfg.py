'''
Quix Config
===========

This module, "Quix Config," provides an extendable framework for constructing
configuration files for PyTorch model training. It uses custom dataclasses where
each represents a group of related configuration parameters, and each field within 
the dataclass corresponds to a specific run parameter.

The module's design emphasizes flexibility and extendability for various training 
configurations in PyTorch. This is achieved by subclassing one of the six config groups:
`RunConfig`, `ModelConfig`, `DataConfig`, `OptimizerConfig`, `AugmentationConfig`,
`SchedulerConfig`, and `LogConfig`. Each group can be extended with dataclass-style
fields and passed to a `RunConfig` for easy construction and parsing of configurations.

Classes
-------
ModelConfig : Base dataclass for model configurations.
DataConfig : Configuration settings for data handling.
OptimizerConfig : Configuration settings for the optimizer.
LogConfig : Configuration settings for logging.
AugmentationConfig : Configuration settings for data augmentation.
SchedulerConfig : Configuration settings for the learning rate scheduler.
RunConfig : Aggregates different configuration aspects into a unified configuration object.

Functions
---------
add_argument : Adds custom arguments to configuration classes.
_parse_docstring : Parses docstrings to extract descriptions.
_extract_metadata : Extracts metadata from class docstrings.
_repr_helper : Helper method for representing nested dataclasses.

Examples
--------
1. **Subclassing a Config Class**:
   Extend `ModelConfig` to add custom arguments for a new model:

   ```python
   from quix.cfg import ModelConfig, add_argument

   class MyModelConfig(ModelConfig):
       """MyModelConfig.

       Attributes
       ----------
       foo : str
           The foo for the model.
       bar : int
           Number of bars for model.
       flag : bool
           Turning on the flag.
       """

       foo: str = "MyPrecious"
       bar: int = 42
       flag: bool = add_argument(default=False, action='store_true')
   ```

2. **Using the Custom Config**:
   Apply your config in `RunConfig`:

   ```python
   runconfig = RunConfig.argparse(modcfg=MyModelConfig)
   ```

3. **Config File Usage**:
   Use a YAML file for configurations:

   ```bash
   python train.py --cfg maincfg.yml
   ```

   For sub-experiments with different parameters:

   ```bash
   python train.py --cfg maincfg.yml exp1.yml
   ```

This module allows for extensive modularity in defining and customizing configurations 
for different PyTorch models or training scenarios, without modifying the original base 
configuration classes. It also supports nested structures in configuration files.

Author
------
Marius Aasan <mariuaas@ifi.uio.no>

Contributors
------------
Please consider contributing to the project.
'''
from __future__ import annotations
import inspect
import warnings
import os
import json
import yaml
from argparse import ArgumentParser, Namespace, ArgumentError, ArgumentTypeError
from numpydoc.docscrape import NumpyDocString
from dataclasses import dataclass, field, fields, is_dataclass, _MISSING_TYPE
from typing import (
    Type, TypeVar, Generic, Any, Dict, Optional, ClassVar, Union,
    Sequence, List, Tuple, get_type_hints, get_args, get_origin
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


def _unified_parser(value:str, types:Sequence[Type], base_types:Sequence[Type], descr:str):
    '''Helper method to parse Union / Base types.
    '''
    for t in types:
        if t in base_types:
            try:
                return t(value)
            except ValueError:
                continue
        elif get_origin(t) is Union:
            return _unified_parser(value, get_args(t), base_types, descr)
    raise ArgumentTypeError(f"Value '{value}' is not valid for {descr}")


def _get_parser(types:Sequence[Type], base_types:Sequence[Type], descr:str):
    '''Factory method for constructing unified parsers.
    '''
    def _parser(x):
        return _unified_parser(x, types, base_types, descr)
    return _parser


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


def _parse_docstring(docstring:str) -> Dict[str, Any]:
    '''Helper method to parse docstrings.
    '''
    doc = NumpyDocString(docstring)
    return {item.name: {'help': '\n'.join(item.desc)} for item in doc['Attributes']}


def _extract_metadata(cls) -> Dict[str, Any]:
    '''Helper function to extract docstring metadata.
    '''
    metadata = {}
    for c in reversed(cls.__mro__):
        cls_docstring = inspect.getdoc(c)
        if cls_docstring:
            metadata.update(_parse_docstring(cls_docstring))
    return metadata


def metadata_decorator(cls):
    '''Metadata decorator.

    This function acts on dataclasses to embed docstring
    descriptions into field metadata. This is later passed to the
    arparser for parsing a config file.

    Parameters
    ----------
    cls
        A class instance for the metatype decorator
    '''
    metadata_dict = _extract_metadata(cls)
    dataclass_fields = {f.name for f in fields(cls)}
    docstring_attr = set(metadata_dict.keys())
    missing_in_docstring = dataclass_fields - docstring_attr
    missing_in_fields = docstring_attr - dataclass_fields

    if missing_in_docstring:
        warnings.warn(
            f"Warning: Fields missing in docstring for {cls.__name__}: {missing_in_docstring}"
        )
    if missing_in_fields:
        warnings.warn(
            f"Warning: Docstring attributes not found as fields in {cls.__name__}: {missing_in_fields}"
        )

    for f in fields(cls):
        if f.name in metadata_dict:
            f.metadata = {'default': f.default, **metadata_dict[f.name], **f.metadata} # type: ignore

    return cls


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
    use_timm_zoo : bool
        Flag to load model from timm (requires timm).
    pretrained_weights : Optional[str]
        Path for pretrained model weights.
    sync_bn : bool
        Sync Batch Norm for DDP.
    '''
    model: str
    use_torch_zoo: bool = True
    use_timm_zoo: bool = add_argument(default=False, action='store_true') # TODO: Make example instead of default
    pretrained_weights: Optional[str] = None
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
    '''
    data_path:str
    dataset:str = 'IN1k'
    workers:int = 16
    prefetch:int = 2
    input_ext:List[str] = add_argument(default=None, nargs='+')
    target_ext:List[str] = add_argument(default=None, nargs='+')
    num_classes:Optional[int] = None


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
    bias_decay : float
        Weight decay for bias.
    norm_decay : float
        Weight decay for norm layers.
    emb_decay : float
        Weight decay for transformer embeddings.
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
    '''
    optim:str = 'adamw'
    lr:float = 3e-3
    weight_decay:float = 2e-2
    bias_decay:float = 0.0
    norm_decay:float = 0.0
    emb_decay:float = 0.0
    opt_epsilon:float = 1e-7
    gradclip:float = 1.0
    accumulation_steps:int = 1
    amsgrad:bool = add_argument(default=False, action='store_true')
    model_ema:bool = add_argument(default=False, action='store_true')
    model_ema_steps:int = 32
    model_ema_decay:float = 0.9998
    amp:bool = add_argument(default=False, action='store_true')
    consistent_batch_size:bool = True


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
    use_neptune : bool
        Flag to use Neptune logging [requires defined API key].
    '''
    savedir:str = './runlogs'
    logfreq:int = 50
    stdout:bool = add_argument(default=False, action='store_true')
    custom_runid:Optional[str] = None
    use_neptune:bool = add_argument(default=False, action='store_true')


class AugmentationConfig(_BaseConfig):
    '''Configuration settings for data augmentation.

    Attributes
    ----------
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


class SchedulerConfig(_BaseConfig):
    '''Configuration settings for the learning rate scheduler.

    Attributes
    ----------
    lr_scheduler : str
        Learning rate scheduler type.
    lr_warmup_epochs : int
        Scheduler warmup epochs.
    lr_min : float
        Learning rate scheduler minimum value.
    lr_init : float
        Learning rate scheduler initial value.
    '''
    lr_scheduler:str = 'cosinedecay'
    lr_warmup_epochs:int = 5
    lr_min:float = 1e-5
    lr_init:float = 1e-6



TMod = TypeVar('TMod', bound=ModelConfig)
TOpt = TypeVar('TOpt', bound=OptimizerConfig)
TDat = TypeVar('TDat', bound=DataConfig)
TLog = TypeVar('TLog', bound=LogConfig)
TAug = TypeVar('TAug', bound=AugmentationConfig)
TSch = TypeVar('TSch', bound=SchedulerConfig)

@metadata_decorator
@dataclass(repr=False)
class RunConfig(Generic[TMod,TDat,TOpt,TLog,TAug,TSch]):
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
    aug : AugmentationConfig
        Augmentation configuration parameters.
    sch : SchedulerConfig
        Scheduler configuration parameters.
    cfg : str
        Path to config file in JSON/YAML format.
    batch_size : int
        Batch size for training. 
    epochs : int
        Number of training epochs. 
    test_only : bool
        Flag for only performing testing / validation.
    device : str
        Device type for training ['cuda' or 'cpu']. 
    use_devices : List[int]
        List of device IDs to use for training.
    ddp_backend : str
        Distributed Data Parallel [DDP] backend. 
    ddp_url : str
        URL for DDP process group. 
    ddp_master_address : str
        IP address for MASTER_ADDR [DDP]. 
    ddp_master_port : str
        Port for MASTER_PORT [DDP]. 
    
    '''
    mod:TMod = add_argument(no_parse=True)
    dat:TDat = add_argument(no_parse=True)
    opt:TOpt = add_argument(no_parse=True)
    log:TLog = add_argument(no_parse=True)
    aug:TAug = add_argument(no_parse=True)
    sch:TSch = add_argument(no_parse=True)
    cfg:str = add_argument(default=[], nargs='*')
    batch_size:int = 2048
    epochs:int = 300
    test_only:bool = add_argument(default=False, action='store_true')
    device:str = add_argument(default='cuda', choices=['cuda', 'cpu'])
    use_devices:List[int] = add_argument(default=[0], nargs='*')
    ddp_backend:str = 'nccl'
    ddp_url:str = 'env://'
    ddp_master_address:str = 'localhost'
    ddp_master_port:str = '29500'

    def __repr__(self):
        return _repr_helper(self)
    
    @classmethod
    def _get_required(
        cls, 
        modcfg:Type[TMod]=ModelConfig,          #type:ignore
        datcfg:Type[TDat]=DataConfig,           #type:ignore
        optcfg:Type[TOpt]=OptimizerConfig,      #type:ignore
        logcfg:Type[TLog]=LogConfig,            #type:ignore
        augcfg:Type[TAug]=AugmentationConfig,   #type:ignore
        schcfg:Type[TSch]=SchedulerConfig,      #type:ignore
    ) -> Sequence[str]:
        # Check if required arguments are provided
        allcfgs = [modcfg, datcfg, optcfg, logcfg, augcfg, schcfg]
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
        augcfg:Type[TAug]=AugmentationConfig,   # type: ignore
        schcfg:Type[TSch]=SchedulerConfig,      # type: ignore
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
        augcfg : Type[AugmentationConfig]
            Dataclass for augmentation configuration.
        schcfg : Type[SchedulerConfig]
            Dataclass for scheduler configuration.

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
        auggrp = parser.add_argument_group('aug')
        schgrp = parser.add_argument_group('sch')
            
        # --- Add Arguments from Config Classes ---
        grppairs = [
            (parser, RunConfig),
            (modgrp, modcfg), (datgrp, datcfg), (optgrp, optcfg), 
            (loggrp, logcfg), (auggrp, augcfg), (schgrp, schcfg)
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
        augcfg:Type[TAug]=AugmentationConfig,   #type:ignore
        schcfg:Type[TSch]=SchedulerConfig,      #type:ignore
    ) -> RunConfig[TMod,TDat,TOpt,TLog,TAug,TSch]:
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
        augcfg : Type[AugmentationConfig]
            Dataclass for augmentation configuration.
        schcfg : Type[SchedulerConfig]
            Dataclass for scheduler configuration.

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

        aug = extract_cfg(augcfg)
        sch = extract_cfg(schcfg)
        mod = extract_cfg(modcfg)
        dat = extract_cfg(datcfg)
        opt = extract_cfg(optcfg)
        log = extract_cfg(logcfg)

        # Combine for RunConfig
        run = extract_cfg(cls, mod=mod, dat=dat, opt=opt, log=log, aug=aug, sch=sch)

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
        augcfg:Type[TAug]=AugmentationConfig,   #type: ignore
        schcfg:Type[TSch]=SchedulerConfig,      #type: ignore
        **kwargs
    ) -> RunConfig[TMod,TDat,TOpt,TLog,TAug,TSch]:
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
        augcfg : Type[AugmentationConfig]
            Dataclass for augmentation configuration.
        schcfg : Type[SchedulerConfig]
            Dataclass for scheduler configuration.
        '''
        allcfgs = [modcfg, datcfg, optcfg, logcfg, augcfg, schcfg]
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


if __name__ == '__main__':
    class MyModelConfig(ModelConfig):
        '''Testing model config.

        Attributes
        ----------
        milkshake : str
            Takes all the boys to the yard.
        '''
        milkshake:str = 'Better than yours'

    runconfig = RunConfig.argparse(modcfg=MyModelConfig)
    runconfig.mod.milkshake
    print(runconfig)
