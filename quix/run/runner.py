from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision as tv
import os
import re
import time
import warnings
import logging
import errno

from torch import Tensor
from contextlib import nullcontext, contextmanager, ExitStack
from torch.utils.data import DataLoader, default_collate, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.swa_utils import AveragedModel
from torch.multiprocessing.spawn import spawn
from typing import Tuple, Sequence, Optional, Dict, Any, ContextManager, Callable, Type, Union
from ..cfg import (
    RunConfig, TMod, TDat, TOpt, TLog,
    ModelConfig, DataConfig, OptimizerConfig, LogConfig
)
from ..proc import BatchProcessor
from ..log import LogCollator
from ..data import QuixDataset, parse_train_augs, parse_val_augs
from ..sched import CosineDecay
from ..ema import ExponentialMovingAverage

'''Overall TODO:

- PRIORITY: Add support for restarts and elastic runs!
- Possibly wrap some logic in hidden calls to simplify logic
    - In particular: Wrap all stuff to do with rank and local_rank to abstract away tedious stuff.
- Fix RASampler
- Fix EMA -> General Model Averaging
- Add AutoAugment
- Fix AugParsing
- Wrapper for Opt-Sched
- Context manager for log_status
'''

TensorSequence = Union[Tensor, Sequence[Tensor]]
CallableContext = Callable[[], ContextManager]

def _getlogger():
    runlog = logging.getLogger('quix.run')
    runlog.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logfmt = logging.Formatter('%(levelname)s:%(asctime)s | %(message)s')
    ch.setFormatter(logfmt)
    runlog.addHandler(ch)
    return runlog

applog = _getlogger()
_dtype_warned = False

def parse_dtype(dtype_str):
    global _dtype_warned
    dtype_map = {
        'float32': torch.float32,
        'float': torch.float32,
        'float64': torch.float64,
        'double': torch.float64,
        'float16': torch.float16,
        'half': torch.float16,
        'bfloat16': torch.bfloat16,
    }

    if dtype_str not in dtype_map and not _dtype_warned:
        applog.warn(f'Invalid dtype: {dtype_str}, defaulting to float32.')
        _dtype_warned = True
    
    return dtype_map.get(dtype_str.lower(), torch.float32)


class AbstractRunner:

    _nimplmsg = '{} misses implementation for {}.'

    def __init__(self, cfg:RunConfig[TMod,TDat,TOpt,TLog]):
        '''Initializes a runner instance.

        Parameters
        ----------
        cfg :RunConfig 
            Parsed RunConfig instance.
        '''

        self.cfg = cfg
        self.distributed = False
        self.input_ext = self.dat.input_ext
        self.target_ext = self.dat.target_ext
        self.world_size = self.rank = self.local_rank = self.local_world_size = None
        torch.set_default_dtype(parse_dtype(self.cfg.dtype))
        distparams = [
            'world_size', 'rank', 'local_world_size', 'local_rank',
            'ddp_master_address', 'ddp_master_port'
        ]
        missing = [k for k in distparams if getattr(cfg, k) is None]

        if missing:
            msg = f'Missing args: {",".join(missing)} for distributed training.\n'
            msg += f'Running with distributed = False'
            self.warningmsg(msg)

        else:
            self.distributed = True
            world_size, rank, local_world_size, local_rank, _, _  = [getattr(cfg, k) for k in distparams]
            torch.cuda.set_device(local_rank)
            current_device = torch.cuda.current_device()
            if self.log.stdout:
                msg = f'Distributed init: {rank=}, {local_rank=}, {world_size=} {current_device=}'
                self.infomsg(msg)
            
            self.world_size, self.rank, self.local_rank, self.local_world_size = (
                world_size, rank, local_rank, local_world_size
            )
            dist.init_process_group(
                backend=cfg.ddp_backend, init_method=cfg.ddp_url, 
                world_size=world_size, rank=rank
            )
            dist.barrier()
    
    @property
    def mod(self):
        return self.cfg.mod
    
    @property
    def dat(self):
        return self.cfg.dat
    
    @property
    def opt(self):
        return self.cfg.opt
    
    @property
    def log(self):
        return self.cfg.log
    
    @property
    def savedir(self):
        order = [self.log.savedir, self.log.project, self.log.custom_runid]
        return os.path.join(*[c for c in order if c is not None])
    
    @property
    def checkpointdir(self):
        return os.path.join(self.savedir, 'checkpoint')
    
    @property
    def local_device(self):
        return (
            f'cuda:{self.local_rank}' 
            if self.local_rank is not None else 'cpu'
        )
    
    @staticmethod
    def combined_context(*context_managers):
        '''This method combines a variable no. context managers.

        Parameters
        ----------
        context_managers : Sequence[Callable[[], ContextManager]]
            Each element should be a callable that returns a context manager.

        Returns
        -------
        A callable of combined context managers.
        '''
        @contextmanager
        def combined_context():
            with ExitStack() as stack:
                for cm in context_managers:
                    stack.enter_context(cm())
                yield  # This allows the with-block to execute

        return combined_context
    
    def debugmsg(self, msg:str):
        if self.distributed:
            if self.rank == 0 and self.local_rank == 0:
                if self.log.stdout:
                    applog.debug(msg)
        else:
            applog.debug(msg)

    def infomsg(self, msg:str):
        if self.distributed:
            if self.rank == 0 and self.local_rank == 0:
                applog.info(msg)
        else:
            applog.info(msg)
    
    def warningmsg(self, msg:str):
        if self.distributed:
            if self.rank == 0 and self.local_rank == 0:
                applog.warning(msg)
        else:
            applog.warning(msg)

    def log_status(self, logger:Optional[LogCollator], **kwargs):
        if logger is not None:
            logger(**kwargs)

    def unpack_data(self, data):
        n_inputs = len(self.input_ext)
        inputs = data[:n_inputs]
        targets = data[n_inputs:]
        return inputs, targets
        
    def send_to_device(self, data):
        if data is not None:
            return tuple(map(
                lambda x: x.to(
                    device=self.local_device,
                    dtype=torch.get_default_dtype()
                ), 
                data
            ))
        return data
    
    def _dataunpacker(self, data):
        inputs, targets = self.unpack_data(data)
        return self.send_to_device(inputs), self.send_to_device(targets)
    
    @staticmethod
    def forward_fn(inputs, targets, model, loss_fn) -> Tuple[Tensor, TensorSequence]:
        raise NotImplementedError('Missing implementation of `forward_fn`.')

    def parse_checkpoint(self, model, optimizer, scheduler, scaler, model_ema) -> int:
        if self.distributed:
            model = model.module

        start_epoch = self.cfg.start_epoch

        if self.mod.resume:
            if not os.path.isfile(self.mod.resume):
                raise FileNotFoundError(f'Invalid checkpoint resume path {self.mod.resume}')
            checkpoint = torch.load(self.mod.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model.to(dtype=torch.get_default_dtype())

            if not self.mod.onlyweights:
                if not self.cfg.test_only:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    if scheduler:
                        scheduler.load_state_dict(checkpoint['scheduler'])

                start_epoch = checkpoint['epoch']

                if model_ema: # TODO: ?
                    model_ema.load_state_dict(checkpoint['model_ema'])

                if scaler:
                    scaler.load_state_dict(checkpoint['scaler'])

        # Initialize checkpoint directory
        os.makedirs(self.checkpointdir, exist_ok=True)
        
        return start_epoch
    
    def parse_augmentations(self) -> Tuple[Tuple[Callable,Callable], Tuple[Callable,Callable]]:
        raise NotImplementedError('Missing implementation of `parse_augmentations`.')
    
    def parse_data(self, augs, collate_fns) -> Tuple[DataLoader, DataLoader, QuixDataset, QuixDataset]:
        raise NotImplementedError('Missing implementation of `parse_data`.')
    
    def parse_model(self) -> nn.Module:
        raise NotImplementedError('Missing implementation of `parse_model`.')
    
    def parse_loss(self) -> Callable:
        raise NotImplementedError('Missing implementation of `parse_loss`.')
    
    def parse_param_groups(self, model) -> Sequence[Tensor]:
        raise NotImplementedError('Missing implementation of `parse_param_groups`.')
    
    def parse_optimizer(self, parameters) -> Optimizer:
        raise NotImplementedError('Missing implementation of `parse_optimizer`.')
    
    def parse_scheduler(self, optimizer, traindata) -> Optional[LRScheduler]:
        raise NotImplementedError('Missing implementation of `parse_scheduler`.')
    
    def parse_scaler(self) -> Optional[GradScaler]:
        raise NotImplementedError('Missing implementation of `parse_scaler`.')
    
    def parse_ddp(self, model) -> nn.Module:
        raise NotImplementedError('Missing implementation of `parse_ddp`.')     
    
    def parse_ema(self, model_without_ddp) -> Optional[AveragedModel]:
        raise NotImplementedError('Missing implementation of `parse_ema`.')     
    
    def parse_logger(self) -> Optional[LogCollator]:
        raise NotImplementedError('Missing implementation of `parse_logger`.')
            
    def parse_run(self):
        self.infomsg('Parsing augmentations...')
        augs, collate_fns = self.parse_augmentations()
        self.infomsg('Parsing data...')
        trainloader, valloader, traindata, valdata = self.parse_data(augs, collate_fns)
        self.infomsg('Parsing model...')
        model = self.parse_model()
        model.to(self.local_device)
        if self.mod.sync_bn and self.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.infomsg('Parsing loss...')
        loss_fn = self.parse_loss()
        self.infomsg('Parsing parameter groups...')
        parameters = self.parse_param_groups(model)
        self.infomsg('Parsing optimizer...')
        optimizer = self.parse_optimizer(parameters)
        self.infomsg('Parsing scaler...')
        scaler = self.parse_scaler()
        self.infomsg('Parsing scheduler...')
        scheduler = self.parse_scheduler(optimizer, trainloader)
        self.infomsg('Parsing DDP...')
        model = self.parse_ddp(model)
        self.infomsg('Parsing EMA...')
        model_ema = self.parse_ema(model)
        self.infomsg('Parsing checkpoint...')
        start_epoch = self.parse_checkpoint(
            model, optimizer, scheduler, scaler, model_ema
        )
        self.infomsg('Parsing logger...')
        logger = self.parse_logger()
        processor = BatchProcessor(
            self.opt.accumulation_steps,
            self.mod.model_ema_steps,
            self.mod.model_ema_warmup_epochs,
            self.opt.gradclip,
            logger,
            self.opt.consistent_batch_size,
            self.cfg.max_step_skipped
        )
        amp_context = autocast if self.cfg.opt.amp else nullcontext
        self.infomsg('Finished parsing!')
        self.log_status(logger, STATUS='RUN_PARSED', cfg=self.cfg.to_json())
        return {
            'trainloader':trainloader,
            'valloader':valloader,
            'model':model,
            'model_ema':model_ema,
            'loss_fn':loss_fn,
            'optimizer':optimizer,
            'scheduler':scheduler,
            'scaler':scaler,
            'processor':processor,
            'start_epoch':start_epoch,
            'train_epoch_context': traindata.shufflecontext if self.dat.shuffle_train else nullcontext,
            'val_epoch_context': nullcontext,
            'train_proc_context': nullcontext,
            'val_proc_context': nullcontext,
            'train_fwd_context': amp_context,
            'val_fwd_context': self.combined_context(amp_context, torch.no_grad),
        }
    
    def load_model_from_checkpoint(self, checkpoint_path):
        model = self.parse_model()
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'Invalid checkpoint resume path {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        return model.to(dtype=torch.get_default_dtype())
    
    @staticmethod
    def _fetch_state_dict(key, **kwargs):
        cls_ = kwargs.get('optimizer', None)
        if cls_ is not None and hasattr(cls_, 'state_dict'):
            return cls_.state_dict()
        return None

    def checkpoint(self, epoch, **run_kwargs):
        '''Method to handle checkpointing.

        For processes with `rank` == `local_rank` == 0, stores a checkpoint.
        Checkpoints include:
            - current epoch
            - model state
            - optimizer state
            - run config
            - scheduler state (if applicable)
            - model ema (if applicable)
            - scaler state (if applicable)

        Parameters
        ----------
        epoch : int
            Current epoch of run.
        run_kwargs : Dict[str, Any]
            The current run arguments. 
        '''        
        model = run_kwargs['model']
        
        if self.distributed:
            model = model.module

        if self.log.savedir:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': self._fetch_state_dict('optimizer', **run_kwargs),
                'cfg': self.cfg.to_dict(),
            }

            if self.opt.lr_scheduler:
                checkpoint['scheduler'] = self._fetch_state_dict('scheduler', **run_kwargs)

            if self.mod.model_ema:
                checkpoint['model_ema'] = self._fetch_state_dict('model_ema', **run_kwargs)

            if self.opt.amp:
                checkpoint['scaler'] = self._fetch_state_dict('scaler', **run_kwargs)

            if (self.distributed and self.rank == 0 and self.local_rank == 0) or not self.distributed:
                savepath_model = os.path.join(self.checkpointdir, f'{self.log.custom_runid}_{epoch:012d}.pth')
                self.infomsg(f'Saving checkpoint: {savepath_model}')
                torch.save(checkpoint, savepath_model)

                if self.log.rolling_checkpoints > 0:
                    # TODO: Verify that this is what we want
                    pattern = rf'{self.log.custom_runid}_\d{{12}}.pth'
                    all_cps = sorted([f for f in os.listdir(self.checkpointdir) if re.match(pattern, f)])
                    for old_cp in all_cps[:-self.log.rolling_checkpoints]:
                        old_cp_path = os.path.join(self.checkpointdir, old_cp)
                        self.infomsg(f'Removing old checkpoint: {old_cp_path}')
                        os.remove(old_cp_path)

    def process_epoch(
        self, 
        epoch:int,
        trainloader:DataLoader,
        valloader:DataLoader, 
        model:nn.Module, 
        model_ema:AveragedModel,
        loss_fn:Callable,
        optimizer:Optimizer,
        scheduler:LRScheduler,
        scaler:GradScaler,
        processor:BatchProcessor, 
        train_epoch_context:CallableContext,
        val_epoch_context:CallableContext,
        train_proc_context:CallableContext,
        val_proc_context:CallableContext,
        train_fwd_context:CallableContext,
        val_fwd_context:CallableContext,
        training:bool=True,
        **logging_kwargs
    ):
        '''Method that handles the processing of a single epoch.

        NOTE: While this can be overridden by subclassing, the intended use
              of inheritance here is to overwrite `forward_fn` and write custom
              parsers for the run. This defines the core logic for both training
              and evaluation.

        Parameters
        ----------
        epoch : int
            Current epoch.
        trainloader : DataLoader
            The dataloader for training.
        valloader : DataLoader, 
            The dataloader for validation.
        model : nn.Module
            The model.
        model_ema : AveragedModel
            The averaged model (currently only EMA).
        loss_fn : Callable
            The loss function for training.
        optimizer : Optimizer
            The optimizer for training.
        scheduler : LRScheduler
            The learning rate scheduler for training.
        scaler : GradScaler
            The gradient scaler for mixed precision.
        processor : BatchProcessor
            A quix batch processor instance.
        train_epoch_context : Callable[[],ContextManager]
            Context for entire epoch in training loop.
        val_epoch_context : Callable[[],ContextManager]
            Context for entire epoch in validation loop.
        train_proc_context : Callable[[],ContextManager]
            Context for processor in training.
        val_proc_context : Callable[[],ContextManager]
            Context for processor in validation.
        train_fwd_context : Callable[[],ContextManager]
            Context for which to run forward function in training.
        val_fwd_context : Callable[[],ContextManager]
            Context for which to run forward function in validation.
        training : bool
            The training state
        logging_kwargs : Dict[str, Any]
            Extra arguments for logging.
        '''
        loader = trainloader if training else valloader
        proc_context = train_proc_context if training else val_proc_context
        epoch_context = train_epoch_context if training else val_epoch_context
        fwd_context = train_fwd_context if training else val_fwd_context
        model.train() if training else model.eval()
        processor_kwargs = {
            'optimizer':optimizer, 'scheduler':scheduler, 'scaler':scaler, 
            'epoch':epoch, 'model':model, 'averaged_model':model_ema, 
            'context':proc_context, 'training':training, **logging_kwargs
        }

        with epoch_context():
            for iteration, data in enumerate(loader):
                final_batch = False
                if hasattr(loader, '__len__'):
                    final_batch = ((len(loader) - 1) == iteration)
                inputs, targets = self._dataunpacker(data)
                current_kwargs = {
                    'iteration': iteration, 'inputs': inputs, 
                    'targets': targets, 'final_batch': final_batch, 
                    'rank': self.rank, 'local_rank': self.local_rank,
                    **processor_kwargs
                }
                with processor(**current_kwargs) as proc:
                    with fwd_context():
                        proc.loss, proc.outputs = self.forward_fn(inputs, targets, model, loss_fn)

                if processor.cancel_run:
                    if self.distributed:
                        distparams = (
                            self.world_size, self.rank, 
                            self.local_world_size, self.local_rank
                        )
                        diststr = ','.join([f'{p=}' for p in distparams])
                    else:
                        diststr = 'non-distributed'

                    self.log_status(processor._logger, STATUS=f'RUN_CANCEL:{diststr.upper()}')
                    raise ValueError(f'Run failure @ {diststr}. Cancelling...')
        return
   
    def run(self):
        run_kwargs = self.parse_run()
        logger = run_kwargs['processor']._logger
        start_epoch = run_kwargs.pop('start_epoch', 0)
        if self.cfg.test_only: # TODO: Make log_staus a context manager
            self.log_status(logger, STATUS='TEST_START')
            self.process_epoch(-1, **run_kwargs, training=False)
            self.log_status(logger, STATUS='TEST_END')
            return
        self.log_status(logger, STATUS='TRAIN_START', epoch='start_epoch')
        for epoch in range(start_epoch, self.cfg.epochs):
            self.process_epoch(epoch, **run_kwargs, training=True)
            self.process_epoch(epoch, **run_kwargs, training=False)
            self.checkpoint(epoch, **run_kwargs)
        self.log_status(logger, STATUS='TRAIN_END', epoch=self.cfg.epochs)
        return
    
    @classmethod
    def argparse(
        cls, 
        mod:Type[TMod]=ModelConfig,       #type:ignore
        dat:Type[TDat]=DataConfig,        #type:ignore
        opt:Type[TOpt]=OptimizerConfig,   #type:ignore
        log:Type[TLog]=LogConfig,         #type:ignore
        **kwargs
    ) -> AbstractRunner:
        '''Parses a Run from dataclasses using argparse.

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
        runconfig = RunConfig.argparse(modcfg=mod,datcfg=dat,optcfg=opt,logcfg=log,**kwargs)
        return cls(runconfig)
    
    @classmethod
    def from_dict(
        cls, 
        mod:Type[TMod]=ModelConfig,       #type:ignore
        dat:Type[TDat]=DataConfig,        #type:ignore
        opt:Type[TOpt]=OptimizerConfig,   #type:ignore
        log:Type[TLog]=LogConfig,         #type:ignore
        **dct
    ):
        '''Parses a Run from dataclasses using a dictionary.

        NOTE: More or less a convenience class for debugging and testing,
              but can be used for notebooks etc.

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
        runconfig = RunConfig.from_dict(modcfg=mod,datcfg=dat,optcfg=opt,logcfg=log,**dct)
        return cls(runconfig)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'  distributed: {self.distributed}\n'
        repr_str += f'  rank: {self.rank}\n'
        repr_str += f'  local_rank: {self.local_rank}\n'
        repr_str += f'  world_size: {self.world_size}\n'
        repr_str += f'  input_ext: {self.input_ext}\n'
        repr_str += f'  target_ext: {self.target_ext}\n'
        for line in self.cfg.__repr__().splitlines():
            repr_str += f'  {line}\n'
        repr_str += ')'
        return repr_str
    

class Runner(AbstractRunner):

    optimizer_dict = { # Fix later
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam
    }

    scheduler_dict = {
        'cosinedecay': CosineDecay
    }

    def __init__(self, cfg:RunConfig[TMod,TDat,TOpt,TLog]):
        super().__init__(cfg)
        self.num_classes = self.cfg.dat.num_classes
        self.input_ext = self.dat.input_ext
        self.target_ext = self.dat.target_ext
        # Yalla yalla
        if self.input_ext is None and self.target_ext is None:
            if self.dat.dataset == 'IN1k':
                self.num_classes = 1000
                self.input_ext = ['jpg']
                self.target_ext = ['cls']


    @staticmethod
    def forward_fn(inputs, targets, model, loss_fn):
        outputs = model(*inputs)
        loss = loss_fn(outputs, *targets)
        return loss, outputs

    def parse_augmentations(self):
        # TODO: Move to this module
        train_sample_augs, train_collate_fn = parse_train_augs(self.dat, num_classes=self.num_classes)
        val_sample_augs, val_collate_fn = parse_val_augs(self.dat)
        return (train_sample_augs, val_sample_augs), (train_collate_fn, val_collate_fn)

    def parse_data(self, augs, collate_fns):
        # Load Augmentations
        train_aug, val_aug = augs
        train_collate, val_collate = collate_fns

        # Load Data
        use_extensions = None
        if self.input_ext is not None or self.target_ext is not None:
            if self.input_ext is not None and self.target_ext is not None:
                use_extensions = self.input_ext + self.target_ext
            elif self.input_ext is not None:
                use_extensions = self.input_ext
            else:
                raise ValueError('Cannot parse target_ext without input_ext!')

        traindata = QuixDataset(
            self.dat.dataset, 
            self.dat.data_path,
            train=True,
            override_extensions=use_extensions,
        )
        valdata = QuixDataset(
            self.dat.dataset,
            self.dat.data_path,
            train=False,
            override_extensions=use_extensions
        )

        # Set shuffle seed
        traindata.set_shuffle_seed(self.dat.shuffle_seed)

        # Map to correct tv_types if applicable
        if traindata.supports_tv_tensor():
            from torchvision import tv_tensors
            from torchvision.transforms import v2

            # Populate dict
            tvt_dct = {
                'image': v2.Compose([
                    v2.ToImage(), 
                    v2.ToDtype(torch.get_default_dtype(), scale=True)
                ]),
                'mask': tv_tensors.Mask,
                'bbox': tv_tensors.BoundingBoxes,
                'other': nn.Identity()
            }
            tv_types = [tvt_dct[d._default_tv] for d in traindata.decoders]
            traindata = traindata.map_tuple(*tv_types)
            valdata = valdata.map_tuple(*tv_types)

        # Map Augmentations
        traindata = traindata.map(train_aug)
        valdata = valdata.map(val_aug)

        # TODO: Add support for RASampler
        if self.distributed:
            trainsampler = DistributedSampler(
                traindata, shuffle=False, drop_last=self.dat.loader_drop_last,
            )
            valsampler = DistributedSampler(
                traindata, shuffle=False,
            )
        else:
            trainsampler = SequentialSampler(traindata)
            valsampler = SequentialSampler(valdata)

        trainloader = DataLoader(
            traindata, 
            self.cfg.batch_size, 
            sampler=trainsampler,
            num_workers=self.dat.workers,
            persistent_workers=True,
            prefetch_factor=self.dat.prefetch,
            pin_memory=True,
            drop_last=self.dat.loader_drop_last,
            collate_fn=train_collate
        )
        valloader = DataLoader(
            valdata, 
            self.cfg.batch_size,
            sampler=valsampler,
            num_workers=self.dat.workers,
            persistent_workers=True,
            prefetch_factor=self.dat.prefetch,
            pin_memory=True,
            collate_fn=val_collate
        )
        return trainloader, valloader, traindata, valdata
    
    def parse_model(self):
        model = tv.models.get_model(self.mod.model, weights=self.mod.pretrained_weights, num_classes=self.num_classes)
        return model
    
    def parse_loss(self):
        return nn.CrossEntropyLoss(label_smoothing=self.cfg.opt.smoothing)
    
    def parse_param_groups(self, model):
        norm_classes = (
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        )
        params = {"other":[], "norm":[]}
        params_weight_decay = {"other": self.cfg.opt.weight_decay, "norm": self.cfg.opt.norm_decay}
        custom_keys = []
        for k, v in zip(self.cfg.opt.custom_decay_keys, self.cfg.opt.custom_decay_vals):
            if v > 0:
                params[k] = []
                params_weight_decay[k] = v
                custom_keys.append(k)

        def _add_params(module, prefix=""):
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                is_custom_key = False
                for key in custom_keys:
                    target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                    if key == target_name:
                        params[key].append(p)
                        is_custom_key = True
                        break
                if not is_custom_key:
                    if self.cfg.opt.norm_decay > 0 and isinstance(module, norm_classes):
                        params["norm"].append(p)
                    else:
                        params["other"].append(p)

            for child_name, child_module in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
                _add_params(child_module, prefix=child_prefix)

        _add_params(model)

        parameters = []
        for key in params:
            if len(params[key]) > 0:
                parameters.append({"params": params[key], "weight_decay": params_weight_decay[key]})

        return parameters
    
    def parse_optimizer(self, parameters) -> Optimizer:
        optcls = self.optimizer_dict.get(self.opt.optim, None)
        if optcls is None:
            raise ValueError(f'Optimizer: {self.opt.optim} not found.')
        if self.opt.zro and self.distributed:
            optimizer = ZeroRedundancyOptimizer(
                parameters, optcls, lr=self.opt.lr, weight_decay=self.opt.weight_decay
            )
        else:
            optimizer = optcls(parameters, lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        return optimizer
    
    def parse_scaler(self) -> Optional[GradScaler]:
        return GradScaler() if self.cfg.opt.amp else None

    def parse_scheduler(self, optimizer:Optimizer, trainloader:DataLoader):
        schcls = self.scheduler_dict.get(self.cfg.opt.lr_scheduler, None)
        if schcls is None:
            raise ValueError(f'Scheduler: {self.cfg.opt.lr_scheduler} not found.')
        try:
            num_steps = len(trainloader)
        except:
            num_steps = None
        return schcls(
            optimizer, self.cfg.opt.lr_init, self.cfg.opt.lr_min, self.cfg.epochs, 
            self.cfg.opt.lr_warmup_epochs / self.cfg.epochs, num_steps
        )
    
    def parse_ddp(self, model):
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank])
        return model
    
    def parse_ema(self, model):
        # TODO: We assume that model ema is connected to distributed... FIX THIS
        if self.distributed:
            model = model.module
        model_ema = None
        if self.cfg.mod.model_ema and self.distributed and self.world_size: 
            adjust = self.world_size * self.cfg.batch_size * self.cfg.mod.model_ema_steps / self.cfg.epochs
            alpha = 1.0 - self.cfg.mod.model_ema_decay
            alpha = min(1.0, alpha*adjust)
            model_ema = ExponentialMovingAverage(model, decay=1-alpha, device=self.local_device)

        return model_ema
    
    def parse_logger(self):
        if (# TODO: Remove this check, handled by savedir
            self.log.custom_runid is not None and 
            self.log.project is not None
        ):
            rank = local_rank = 0
            if self.distributed: # TODO: Wrap these in hidden call to simplify logic
                if self.rank is not None:
                    rank = self.rank
                if self.local_rank is not None:
                    local_rank = self.local_rank
            return LogCollator.standard_logger(
                self.log.custom_runid, 
                self.savedir,
                rank,
                local_rank,
                stdout=self.log.stdout
            )
        return None


def __worker(rank, runnercls:Type[Runner], cfgdict):
    cfgdict['rank'] = rank
    cfgdict['local_rank'] = rank
    runnercls.from_dict(**cfgdict).run()


def single_node_launcher(runnercls:Type[Runner], **cfgdict):
    '''Function to launch single node multi-gpu training.

    NOTE: The keyword arguments passed to this function are parsed
          as a config file using the runner's `from_dict` 
          instantiation method. 

    Parameters
    ----------
    runnercls : Type[Runner]
        A subclass of Runner for which to launch the training.
    '''
    world_size = torch.cuda.device_count()
    cfgdict['world_size'] = world_size
    cfgdict['local_world_size'] = world_size
    spawn(__worker, (runnercls, cfgdict,), nprocs=world_size, join=True)
