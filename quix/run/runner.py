from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision as tv
import os
import time
import warnings
import errno

from torch import Tensor
from contextlib import nullcontext
from torch.utils.data import DataLoader, default_collate, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.swa_utils import AveragedModel
from typing import Tuple, Sequence, Optional, Dict, Any, ContextManager, Callable, Type, Union
from ..cfg import (
    RunConfig, TMod, TDat, TOpt, TLog,
    ModelConfig, DataConfig, OptimizerConfig, LogConfig
)
from ..proc import BatchProcessor
from ..log import BaseLogHandler
from ..data import QuixDataset, parse_train_augs, parse_val_augs
from ..sched import CosineDecay
from ..ema import ExponentialMovingAverage

TensorSequence = Union[Tensor, Sequence[Tensor]]
CallableContext = Callable[[], ContextManager]

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
        self.world_size = self.rank = self.local_rank = None
        distparams = ['world_size', 'rank', 'local_rank']
        missing = [k for k in distparams if not getattr(cfg, k)]

        if missing:
            msg = f'Missing args: {",".join(missing)} for distributed training.\n'
            msg += f'Running with distributed = False'
            if cfg.log.stdout:
                print(msg)
            warnings.warn(msg)

        else:
            self.distributed = True
            world_size, rank, local_rank = [getattr(cfg, k) for k in distparams]
            torch.cuda.set_device(local_rank)
            if cfg.log.stdout:
                msg = f'Distributed init: {rank=}, {local_rank=}, {world_size=}'
            
            self.world_size, self.rank, self.local_rank = world_size, rank, local_rank
            dist.init_process_group(
                backend=cfg.ddp_backend, init_method=cfg.ddp_url, 
                world_size=world_size, rank=rank
            )
            dist.barrier()
            self.set_distprint(rank == 0)

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

    @staticmethod
    def set_distprint(is_master:bool) -> None:
        """Disables printing when not in master process

        From https://github.com/pytorch/vision/blob/main/references/classification/utils.py
        """
        import builtins as __builtin__

        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop("force", False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    @staticmethod
    def forward_fn(inputs, targets, model, loss_fn) -> Tuple[Tensor, TensorSequence]:
        raise NotImplementedError('Missing implementation of `forward_fn`.')
    
    def parse_augmentations(self) -> Sequence[Callable]:
        raise NotImplementedError('Missing implementation of `parse_augmentations`.')
    
    def parse_data(self) -> Tuple[DataLoader, DataLoader, QuixDataset, QuixDataset]:
        raise NotImplementedError('Missing implementation of `parse_data`.')
    
    def parse_model(self) -> nn.Module:
        raise NotImplementedError('Missing implementation of `parse_model`.')
    
    def parse_loss(self) -> Callable:
        raise NotImplementedError('Missing implementation of `parse_loss`.')
    
    def parse_param_groups(self, model) -> Sequence[Tensor]:
        raise NotImplementedError('Missing implementation of `parse_param_groups`.')
    
    def parse_optimizer(self, model) -> Optimizer:
        raise NotImplementedError('Missing implementation of `parse_optimizer`.')
    
    def parse_scheduler(self, optimizer, traindata) -> Optional[LRScheduler]:
        raise NotImplementedError('Missing implementation of `parse_scheduler`.')
    
    def parse_scaler(self) -> Optional[GradScaler]:
        raise NotImplementedError('Missing implementation of `parse_scaler`.')
    
    def parse_ddp(self, model) -> Sequence[nn.Module]:
        raise NotImplementedError('Missing implementation of `parse_ddp`.')     
    
    def parse_ema(self, model_without_ddp) -> Optional[AveragedModel]:
        raise NotImplementedError('Missing implementation of `parse_ema`.')     
    
    def parse_checkpoint(self, model, optimizer, scheduler, scaler, model_ema) -> None:
        raise NotImplementedError('Missing implementation of `parse_checkpoint`.')

    def parse_logger(self, start_epoch) -> Optional[BaseLogHandler]:
        raise NotImplementedError('Missing implementation of `parse_logger`.')
        
    def parse_run(self):
        trainloader, valloader, traindata, valdata = self.parse_data()
        model = self.parse_model()
        loss_fn = self.parse_loss()
        optimizer = self.parse_optimizer(model)
        scaler = self.parse_scaler()
        scheduler = self.parse_scheduler(optimizer, traindata)
        model, model_without_ddp = self.parse_ddp(model)
        model_ema = self.parse_ema(model_without_ddp)
        start_epoch = self.parse_checkpoint(
            model_without_ddp, optimizer, scheduler, scaler, model_ema
        )
        logger = self.parse_logger(start_epoch)
        processor = BatchProcessor(
            self.opt.accumulation_steps,
            self.mod.model_ema_steps,
            self.mod.model_ema_warmup_epochs,
            self.opt.gradclip,
            logger,
            self.opt.consistent_batch_size
        )
        return {
            'trainloader':trainloader,
            'valloader':valloader,
            'model':model,
            'model_without_ddp': model_without_ddp,
            'model_ema':model_ema,
            'loss_fn':loss_fn,
            'optimizer':optimizer,
            'scheduler':scheduler,
            'scaler':scaler,
            'processor':processor,
            'start_epoch':start_epoch,
            'epoch_context': traindata.shufflecontext,
            'proc_context': nullcontext,
            'fwd_context': autocast if self.cfg.opt.amp else nullcontext,
        }


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
        epoch_context:CallableContext,
        proc_context:CallableContext,
        fwd_context:CallableContext,
        training:bool=True,
        **logging_kwargs
    ):
        loader = trainloader if training else valloader
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
                n_inputs = len(self.cfg.dat.input_ext)
                inputs, targets = data[:n_inputs], data[n_inputs:]
                kwargs = {
                    'iteration': iteration, 'inputs': inputs, 
                    'targets': targets, 'final_batch':final_batch, 
                    **processor_kwargs
                }
                with processor(**kwargs) as proc:
                    with fwd_context():
                        proc.loss, proc.outputs = self.forward_fn(inputs, targets, model, loss_fn)
        return
   
    def run(self):
        run_kwargs = self.parse_run()
        start_epoch = run_kwargs.pop('start_epoch', 0)
        model_without_ddp = run_kwargs.pop('model_without_ddp', None)
        if self.cfg.test_only:
            self.process_epoch(-1, **run_kwargs, training=False)
            return
        for epoch in range(start_epoch, self.cfg.epochs):
            self.process_epoch(epoch, **run_kwargs, training=True)
            self.process_epoch(epoch, **run_kwargs, training=False)
            if self.log.savedir:
                checkpoint = {
                    'model': model_without_ddp,
                    'optimizer': run_kwargs.get('optimizer', None),
                    'scheduler': run_kwargs.get('scheduler', None),
                    'epoch': epoch,
                    'cfg': self.cfg,
                }
                if self.mod.model_ema:
                    checkpoint['model_ema'] = run_kwargs.get('model_ema', None)
                if self.opt.amp:
                    checkpoint['scaler'] = run_kwargs.get('scaler', None)
                if self.rank == 0:
                    savepath_model = os.path.join(self.log.savedir, f'{self.log.custom_runid}_{epoch:08d}.pth')
                    torch.save(checkpoint, savepath_model) # TODO: Rolling save
        return

    @classmethod
    def argparse(
        cls, 
        mod:TMod=ModelConfig,       #type:ignore
        dat:TDat=DataConfig,        #type:ignore
        opt:TOpt=OptimizerConfig,   #type:ignore
        log:TLog=LogConfig,         #type:ignore
        **kwargs
    ) -> AbstractRunner:
        '''Parses a Run from dataclasses

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
        runconfig = RunConfig.argparse(mod=mod,dat=dat,opt=opt,log=log,**kwargs)
        return cls(runconfig)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'  distributed: {self.distributed}\n'
        repr_str += f'  rank: {self.rank}\n'
        repr_str += f'  local_rank: {self.local_rank}\n'
        repr_str += f'  world_size: {self.world_size}\n'
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
        if self.dat.dataset == 'IN1k':
            self.num_classes = 1000

    @staticmethod
    def forward_fn(inputs, targets, model, loss_fn):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        return loss, outputs

    def parse_augmentations(self):
        train_sample_augs, train_batch_augs = parse_train_augs(self.dat, num_classes=self.num_classes)
        val_sample_augs = parse_val_augs(self.dat)
        return train_sample_augs, train_batch_augs, val_sample_augs

    def parse_data(self):
        # Load Augmentations
        train_sample_augs, train_batch_augs, val_sample_augs = self.parse_augmentations()

        # Load Data
        use_extensions = None
        if self.dat.input_ext is not None or self.dat.target_ext is not None:
            if self.dat.input_ext is not None and self.dat.target_ext is not None:
                use_extensions = self.dat.input_ext + self.dat.target_ext
            else:
                raise ValueError('Both input_ext and target_ext must be set.')

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

        # Map to correct tv_types if applicable
        if traindata.supports_tv_tensor():
            from torchvision import tv_tensors
            from torchvision.transforms import v2

            # Populate dict
            tvt_dct = {
                'image': v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                'mask': tv_tensors.Mask,
                'bbox': tv_tensors.BoundingBoxes,
                'other': nn.Identity()
            }        
            tv_types = [tvt_dct[d._default_tv] for d in traindata.decoders]
            traindata = traindata.map_tuple(*tv_types)
            valdata = valdata.map_tuple(*tv_types)

        # Map Augmentations
        traindata = traindata.map(train_sample_augs)
        valdata = valdata.map(val_sample_augs)

        # Define default collate function
        def train_collate_fn(batch):
            return train_batch_augs(*default_collate(batch))

        # TODO: Add support for RASampler
        sampler = DistributedSampler if self.distributed else SequentialSampler
        trainloader = DataLoader(
            traindata, 
            self.cfg.batch_size, 
            sampler=sampler(traindata),
            num_workers=self.dat.workers,
            prefetch_factor=self.dat.prefetch,
            pin_memory=True,
            collate_fn=train_collate_fn
        )
        valloader = DataLoader(
            valdata, 
            self.cfg.batch_size,
            sampler=sampler(valdata),
            num_workers=self.dat.workers,
            prefetch_factor=self.dat.prefetch,
            pin_memory=True,
            collate_fn=default_collate
        )
        return trainloader, valloader, traindata, valdata
    
    def parse_model(self):
        # Create Model
        model = tv.models.get_model(self.mod.model, weights=self.mod.pretrained_weights, num_classes=self.num_classes)
        model.to(self.cfg.device)
        if self.mod.sync_bn and self.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model
    
    def parse_loss(self):
        return nn.CrossEntropyLoss(label_smoothing=self.cfg.opt.smoothing)
    
    def parse_param_groups(self, model):
        # Set up weight decay groups
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
    
    def parse_optimizer(self, model:nn.Module) -> Optimizer:
        parameters = self.parse_param_groups(model)
        optcls = self.optimizer_dict.get(self.opt.optim, None)
        if optcls is None:
            raise ValueError(f'Optimizer: {self.opt.optim} not found.')
        optimizer = optcls(parameters, lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        return optimizer
    
    def parse_scaler(self) -> Optional[GradScaler]:
        return GradScaler() if self.cfg.opt.amp else None

    def parse_scheduler(self, optimizer:Optimizer, traindata:QuixDataset):
        schcls = self.scheduler_dict.get(self.cfg.opt.lr_scheduler, None)
        if schcls is None:
            raise ValueError(f'Scheduler: {self.cfg.opt.lr_scheduler} not found.')        
        return schcls(
            optimizer, self.cfg.opt.lr_init, self.cfg.opt.lr_min, self.cfg.epochs, 
            self.cfg.opt.lr_warmup_epochs / self.cfg.epochs, self.cfg.batch_size, len(traindata)
        )
    
    def parse_ddp(self, model):
        model_without_ddp = model
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank])
            model_without_ddp = model.module
        return model, model_without_ddp
    
    def parse_ema(self, model_without_ddp):
        model_ema = None
        if self.cfg.mod.model_ema and self.distributed and self.world_size: 
            adjust = self.world_size * self.cfg.batch_size * self.cfg.mod.model_ema_steps / self.cfg.epochs
            alpha = 1.0 - self.cfg.mod.model_ema_decay
            alpha = min(1.0, alpha*adjust)
            model_ema = ExponentialMovingAverage(model_without_ddp, decay=1-alpha, device=self.cfg.device)

        return model_ema
    
    def parse_checkpoint(self, model_without_ddp, optimizer, scheduler, scaler, model_ema):
        start_epoch = self.cfg.start_epoch
        if self.cfg.mod.resume:
            checkpoint = torch.load(self.cfg.mod.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not self.cfg.test_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['start_epoch']
            if model_ema:
                model_ema.load_state_dict(checkpoint['model_ema'])
            if scaler:
                scaler.load_state_dict(checkpoint['scaler'])
        return start_epoch    



