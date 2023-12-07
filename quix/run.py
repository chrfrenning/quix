import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision as tv
import os
import time
import errno

from torch import Tensor
from contextlib import nullcontext
from torch.utils.data import DataLoader, default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from dataclasses import dataclass, field
from typing import Tuple, Sequence, Optional, Dict, Any, ContextManager, Callable, Type
from .cfg import RunConfig, _RunConfig, TMod,TOpt,TAug,TDat,TLog,TSch
from .proc import BatchProcessor
from .log import BaseLogHandler
from .data import QuixDataset, parse_train_augs, parse_val_augs
from .sched import CosineDecay


'''
TODO: 
- Expand main script to DDP
- Parts:
    Init distributed framework and initialize logging folder.
        - Main / Run init
    Parse augmentations.
    Parse data.
    Parse model.
    Parse optimizer
        Init loss function.
        Setup parameter groups
        Init optimizer.
        Init scaler
        Init scheduler
        Init EMA
    Setup DDP model
    Load checkpoints, if existing

Each of these needs to be added to the parsing pipeline.
Need to setup robust loading of distributed params from os.environ, including slurm.

Finally:
    Init BatchProcessor
    Init TrainLoop / ValLoop
    Run train / eval using all components.
'''
def main(cfg:RunConfig):
    # Create logging output if it doesn't exist
    if cfg.log.savedir is not None:
        os.makedirs(cfg.log.savedir, exist_ok=True)
    
    #utils.init_distributed_model(cfg)
    if cfg.log.stdout:
        print(cfg)

    if cfg.device == 'cuda':
        main_device = torch.device(f'cuda:{cfg.use_devices[0]}')
    else:
        main_device = torch.device('cpu')
    
    # Load Augmentations
    train_sample_augs, train_batch_augs = parse_train_augs(cfg.aug)
    val_sample_augs = parse_val_augs(cfg.aug)

    # Load Data
    use_extensions = None
    if cfg.dat.input_ext is not None or cfg.dat.target_ext is not None:
        if cfg.dat.input_ext is not None and cfg.dat.target_ext is not None:
            use_extensions = cfg.dat.input_ext + cfg.dat.target_ext
        else:
            raise ValueError('Both input_ext and target_ext must be set.')

    traindata = QuixDataset(
        cfg.dat.dataset, 
        cfg.dat.data_path,
        train=True,
        override_extensions=use_extensions,
    )
    valdata = QuixDataset(
        cfg.dat.dataset,
        cfg.dat.data_path,
        train=False,
        override_extensions=use_extensions
    )

    num_classes = cfg.dat.num_classes
    if cfg.dat.dataset == 'IN1k':
        num_classes = 1000

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
        cfg.batch_size, 
        sampler=None,
        num_workers=cfg.dat.workers,
        prefetch_factor=cfg.dat.prefetch,
        pin_memory=True,
        collate_fn=train_collate_fn
    )
    valloader = DataLoader(
        valdata, 
        cfg.batch_size,
        sampler=None,
        num_workers=cfg.dat.workers,
        prefetch_factor=cfg.dat.prefetch,
        pin_memory=True,
        collate_fn=default_collate
    )

    # Create Model
    model = tv.models.get_model(cfg.mod.model, weights=cfg.mod.pretrained_weights, num_classes=num_classes)
    model.to(main_device)

    if cfg.mod.sync_bn: # Add test for distributed...
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Simple example for now
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.opt.smoothing)

    # Set up weight decay
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )
    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": cfg.opt.weight_decay,
        "norm": cfg.opt.norm_decay,
    }
    custom_keys = []
    for k, v in zip(cfg.opt.custom_decay_keys, cfg.opt.custom_decay_vals):
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
                if cfg.opt.norm_decay > 0 and isinstance(module, norm_classes):
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

    # Parsing optimizer, just an example
    if cfg.opt.optim not in ['adamw']:
        raise ValueError('Yeah, there is no other option yet.')
    
    optimizer = torch.optim.AdamW(parameters, lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)

    # Parsing scaler for amp
    scaler = torch.cuda.amp.grad_scaler.GradScaler() if cfg.opt.amp else None

    # Initialize LR Scheduler
    if cfg.sch.lr_scheduler not in ['cosinedecay']:
        raise ValueError('Yeah, there is no other option yet.')
    
    scheduler = CosineDecay(
        optimizer, cfg.sch.lr_init, cfg.sch.lr_min, cfg.epochs, 
        cfg.sch.lr_warmup_epochs / cfg.epochs, cfg.batch_size, len(traindata)
    )

    # Init model as DDP
    model_without_ddp = model
    if False: # Add flag for distributed
        model = nn.parallel.DistributedDataParallel(model, device_ids=None) # Add device ids
        model_without_ddp = model.module

    # Init model EMA
    model_ema = None
    if cfg.opt.model_ema and False: # Add distributed params
        adjust = world_size * cfg.batch_size * cfg.opt.model_ema_steps / args.epochs
        alpha = 1.0 - cfg.opt.model_ema_decay
        alpha = min(1.0, alpha*adjust)
        model_ema = None # Add EMA class

    # Resume run from checkpoint, if applicable
    start_epoch = cfg.start_epoch
    if cfg.mod.resume:
        checkpoint = torch.load(cfg.mod.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not cfg.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['start_epoch']
        if model_ema:
            model_ema.load_state_dict(checkpoint['model_ema'])
        if scaler:
            scaler.load_state_dict(checkpoint['scaler'])

    # Check for test only
    if cfg.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            # TODO: Run evaluation with EMA
            pass
        else:
            # TODO: Run evaluation without EMA
            pass
        return

    start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        if False: # Add check for distributed
            train_sampler.set_epoch(epoch)
        # Train one epoch function
        if model_ema:
            # TODO: Run evaluation with EMA
            pass
        else:
            # TODO: Run evaluation without EMA
            pass

        if cfg.log.savedir:
            # Do log in checkpoint dictionary.
            # Include EMA / scaler if applicable
            pass
        
        # Save on master trick here
    
    total_time = time.time() - start_time
    # Print total time

    return traindata, trainloader, valdata, valloader


class BatchTrainer:

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[], Tuple[Tensor, Sequence[Tensor]]],
        forward_fn: Callable,
        context: ContextManager = nullcontext()
        ):

        self.model = model
        self.context = context
        self.forward_fn = forward_fn
        self.loss_fn = loss

    def __call__(self, inputs:Sequence[Tensor], targets:Sequence[Tensor]) -> Tuple[Tensor, Sequence[Tensor]]:
        with self.context:
            loss, outputs = self.forward_fn(inputs, targets, self.model, self.loss_fn)
        
        return loss, outputs
    

class AbstractRunWorker:

    __nimpmsg = "AbstractRun does not have implementation for '{}'"

    def __init__(self, cfg:RunConfig):
        '''AbstractRun class.

        Args:
            config (RunConfig): Parsed RunConfig instance.
        '''
        self.cfg:RunConfig = cfg

    def parse_dataloaders(self, rank:int) -> Sequence[DataLoader]:
        raise NotImplementedError(self.__nimpmsg.format("parse_dataloaders"))

    def parse_model(self, rank:int) -> nn.Module:
        raise NotImplementedError(self.__nimpmsg.format("parse_model"))
    
    def parse_loss(self, rank:int) -> Callable:
        raise NotImplementedError(self.__nimpmsg.format("parse_loss"))
    
    def parse_fwd(self, rank:int) -> Callable:
        raise NotImplementedError(self.__nimpmsg.format("parse_fwd"))
    
    def parse_fwdctx(self, rank:int) -> ContextManager:
        return nullcontext()
    
    def parse_optimizer(self, rank:int, model:nn.Module) -> Optimizer:
        raise NotImplementedError(self.__nimpmsg.format("parse_optimizer"))
    
    def parse_scheduler(self, rank:int, optimizer:Optimizer) -> Optional[LRScheduler]:
        return None

    def parse_scaler(self, rank:int) -> Optional[GradScaler]:
        return None
    
    def parse_logger(self, rank:int) -> Optional[BaseLogHandler]:
        return None

    def parse(self, rank:int) -> Tuple[DataLoader, DataLoader, BatchTrainer, BatchProcessor]:
        trainloader, valloader = self.parse_dataloaders(rank)

        model = self.parse_model(rank)
        loss_fn = self.parse_loss(rank)
        fwd_fn = self.parse_fwd(rank)
        fwdctx = self.parse_fwdctx(rank)
        trainer = BatchTrainer(model, loss_fn, fwd_fn, fwdctx)

        optimizer = self.parse_optimizer(rank, model)
        scheduler = self.parse_scheduler(rank, optimizer)
        scaler = self.parse_scaler(rank)
        logger = self.parse_logger(rank)
        processor = BatchProcessor(
            optimizer, scheduler, scaler, 
            self.cfg.opt.accumulation_steps,
            self.cfg.opt.gradclip,
            logger, self.cfg.opt.consistent_batch_size
        )

        return trainloader, valloader, trainer, processor


    def split_inputs_and_targets(self, data:Sequence[Tensor]):
        if len(data) > self.cfg.data_maxlen:
            raise ValueError('Length of data exceeds max length {} from target indices.')
        targets = [data[i] for i in self.cfg.targetindices]
        inputs = [data[i] for i in range(len(data)) if i not in self.cfg.targetindices]
        return inputs, targets
   
    def launch(self, rank:int):
        if self.cfg.use_ddp:
            dist.init_process_group(
                # TODO: Fix
                backend='gloo',
                init_method='env://',
                world_size=self.cfg.world_size,
                rank=rank
            )
        trainloader, valloader, trainer, processor = self.parse(rank)

        for e in range(self.cfg.epochs):

            for it, data in trainloader:
                final_batch = False
                if hasattr(trainloader, '__len__'):
                    final_batch = ((len(trainloader) - 1) == it)
                inputs, targets = self.split_inputs_and_targets(data)
                with processor(e, it, inputs, targets, final_batch, nullcontext()) as train:
                    train.loss, train.outputs = trainer(inputs, targets)

            for it, data in valloader:
                final_batch = False
                if hasattr(trainloader, '__len__'):
                    final_batch = ((len(trainloader) - 1) == it)
                inputs, targets =  self.split_inputs_and_targets(data)
                with processor(e, it, inputs, targets, final_batch, nullcontext(), False) as val:
                    val.loss, val.outputs = trainer(inputs, targets)


def main_worker(rank:int, runworker:AbstractRunWorker):
    '''Main worker for run class.

    NOTE: This needs to be defined as a top level function in a module, and take rank as the first argument.

    Args:
        rank (int): Process rank.
        runworker (AbstractRunWorker): A custom implementation of a RunWorker.
    '''
    runworker.launch(rank)


class RunHandler:

    def __init__(
        self, 
        runworker:Type[AbstractRunWorker],

        **_kwargs
    ):
        '''Parses and stores the parameters of a run.
        '''
        self.RunWorker:Type[AbstractRunWorker] = runworker

    def _getmaxlen(self, targetindices):
        max_positive = max((index for index in targetindices if index >= 0), default=-1)
        max_negative = min((index for index in targetindices if index < 0), default=0)
        length_for_positive = max_positive + 1
        length_for_negative = abs(max_negative)
        return max(length_for_positive, length_for_negative)
    
    def __call__(self):
        runworker = self.RunWorker(self.cfg)
        if not self.cfg.use_ddp:
            # Do not use ddp
            runworker.launch(0)
        
        else:
            # TODO: Fix
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            mp.spawn(main_worker, (runworker,), nprocs=self.cfg.world_size) # type:ignore

