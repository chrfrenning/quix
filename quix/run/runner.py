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
from torch.utils.data import DataLoader, default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from typing import Tuple, Sequence, Optional, Dict, Any, ContextManager, Callable, Type
from ..cfg import (
    RunConfig, TMod, TDat, TOpt, TLog,
    ModelConfig, DataConfig, OptimizerConfig, LogConfig
)
from ..proc import BatchProcessor
from ..log import BaseLogHandler
from ..data import QuixDataset, parse_train_augs, parse_val_augs
from ..sched import CosineDecay


class Runner:

    optimizer_dict = { # Fix later
        'adamw': torch.optim.AdamW,
    }

    scheduler_dict = {
        'cosinedecay': CosineDecay
    }

    def __init__(self, cfg:RunConfig[TMod,TDat,TOpt,TLog]):
        '''Run class.

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
            return
        
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
        self.setup_for_distributed(rank == 0)

    @staticmethod
    def setup_for_distributed(is_master:bool) -> None:
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
    def forward_fn(inputs, targets, model, loss_fn):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        return loss, outputs

    def parse_augmentations(self):
        train_sample_augs, train_batch_augs = parse_train_augs(self.cfg.dat)
        val_sample_augs = parse_val_augs(self.cfg.dat)
        return train_sample_augs, train_batch_augs, val_sample_augs

    def parse_data(self):
        # Load Augmentations
        train_sample_augs, train_batch_augs, val_sample_augs = self.parse_augmentations()

        # Load Data
        use_extensions = None
        if self.cfg.dat.input_ext is not None or self.cfg.dat.target_ext is not None:
            if self.cfg.dat.input_ext is not None and self.cfg.dat.target_ext is not None:
                use_extensions = self.cfg.dat.input_ext + self.cfg.dat.target_ext
            else:
                raise ValueError('Both input_ext and target_ext must be set.')

        traindata = QuixDataset(
            self.cfg.dat.dataset, 
            self.cfg.dat.data_path,
            train=True,
            override_extensions=use_extensions,
        )
        valdata = QuixDataset(
            self.cfg.dat.dataset,
            self.cfg.dat.data_path,
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

        # TODO: Fix samplers!
        trainloader = DataLoader(
            traindata, 
            self.cfg.batch_size, 
            sampler=None,
            num_workers=self.cfg.dat.workers,
            prefetch_factor=self.cfg.dat.prefetch,
            pin_memory=True,
            collate_fn=train_collate_fn
        )
        valloader = DataLoader(
            valdata, 
            self.cfg.batch_size,
            sampler=None,
            num_workers=self.cfg.dat.workers,
            prefetch_factor=self.cfg.dat.prefetch,
            pin_memory=True,
            collate_fn=default_collate
        )
        return trainloader, valloader, traindata, valdata
    
    def parse_model(self):
        num_classes = self.cfg.dat.num_classes
        if self.cfg.dat.dataset == 'IN1k':
            num_classes = 1000        

        # Create Model
        model = tv.models.get_model(self.cfg.mod.model, weights=self.cfg.mod.pretrained_weights, num_classes=num_classes)
        model.to(self.cfg.device)

        if self.cfg.mod.sync_bn: # Add test for distributed...
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return model

    def parse_optimization(self, model:nn.Module, n_samples:Optional[int]=None):
        # Get loss function
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.cfg.opt.smoothing)

        # Set up weight decay
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

        # Parsing optimizer
        optcls = self.optimizer_dict.get(self.cfg.opt.optim, None)
        if optcls is None:
            raise ValueError(f'Optimizer: {self.cfg.opt.optim} not found.')
        optimizer = optcls(parameters, lr=self.cfg.opt.lr, weight_decay=self.cfg.opt.weight_decay)

        # Parsing scaler for amp
        scaler = GradScaler() if self.cfg.opt.amp else None

        # Initialize LR Scheduler
        schcls = self.scheduler_dict.get(self.cfg.opt.lr_scheduler, None)
        if schcls is None:
            raise ValueError(f'Scheduler: {self.cfg.opt.lr_scheduler} not found.')        
        scheduler = schcls(
            optimizer, self.cfg.opt.lr_init, self.cfg.opt.lr_min, self.cfg.epochs, 
            self.cfg.opt.lr_warmup_epochs / self.cfg.epochs, self.cfg.batch_size, n_samples
        )

        # Init model as DDP
        model_without_ddp = model
        if self.distributed: # Add flag for distributed
            model = nn.parallel.DistributedDataParallel(model, device_ids=None) # Add device ids
            model_without_ddp = model.module

        # Init model EMA
        model_ema = None
        if self.cfg.opt.model_ema and self.distributed and self.world_size: 
            adjust = self.world_size * self.cfg.batch_size * self.cfg.opt.model_ema_steps / self.cfg.epochs
            alpha = 1.0 - self.cfg.opt.model_ema_decay
            alpha = min(1.0, alpha*adjust)
            model_ema = None # Add EMA class

        # Resume run from checkpoint, if applicable
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

        return loss_fn, optimizer, scaler, scheduler, model_ema, start_epoch
    

    def parse_run(self):
        trainloader, valloader, traindata, valdata = self.parse_data()
        model = self.parse_model()
        loss_fn, optimizer, scaler, scheduler, model_ema, start_epoch = self.parse_optimization(model, len(traindata))
        processor = BatchProcessor(
            optimizer, scheduler, scaler, 
            self.cfg.opt.accumulation_steps, 
            self.cfg.opt.gradclip,
            None, # TODO: Missing LogHandler
            self.cfg.opt.consistent_batch_size
        )
        return {
            'trainloader':trainloader,
            'valloader':valloader,
            'model':model,
            'loss_fn':loss_fn,
            'processor':processor,
            'model_ema':model_ema,
            'start_epoch':start_epoch,
            'epoch_context': traindata.shufflecontext,
            'proc_context': nullcontext,
            'fwd_context': autocast if self.cfg.opt.amp else nullcontext,
        }
    
    def process_epoch(
        self, 
        epoch,
        loader, 
        model, 
        loss_fn,
        processor, 
        epoch_context,
        proc_context,
        fwd_context,
        train=True,
        **logging_kwargs
    ):
        with epoch_context():
            for iteration, data in enumerate(loader):
                final_batch = False
                if hasattr(loader, '__len__'):
                    final_batch = ((len(loader) - 1) == iteration)
                n_inputs = len(self.cfg.dat.input_ext)
                inputs, targets = data[:n_inputs], data[n_inputs:]
                with processor(epoch, iteration, inputs, targets, final_batch, proc_context, train) as train:
                    with fwd_context():
                        train.loss, train.outputs = self.forward_fn(inputs, targets, model, loss_fn)
        return
   
    def run(self):
        run_kwargs = self.parse_run()
        if self.cfg.test_only:
            self.process_epoch(-1, **run_kwargs, train=False)
            return
        for epoch in range(self.cfg.epochs):
            self.process_epoch(epoch, **run_kwargs, train=True)
            self.process_epoch(epoch, **run_kwargs, train=False)
        return

    @classmethod
    def argparse(
        cls, 
        mod:TMod=ModelConfig,       #type:ignore
        dat:TDat=DataConfig,        #type:ignore
        opt:TOpt=OptimizerConfig,   #type:ignore
        log:TLog=LogConfig,         #type:ignore
        **kwargs
    ) -> Runner:
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

