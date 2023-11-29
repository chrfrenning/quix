import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from torch import Tensor
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from dataclasses import dataclass, field
from typing import Tuple, Sequence, Optional, Dict, Any, ContextManager, Callable, Type
from .opt import BatchProcessor
from .log import BaseLogHandler


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
    

@dataclass
class _AbstractConfigContainer:

    _additional_attr:Dict[str, Any] = field(init=False)

    def __post_init__(self):
        self._additional_attr = {}

    def add_attr(self, **kwargs) -> None:
        self._additional_attr = {**self._additional_attr, **kwargs}
    
    def __getattr__(self, __name: str) -> Any:
        if __name in self._additional_attr:
            return self._additional_attr[__name]
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute '{__name}'"
        )

@dataclass
class _AugConfigContainer(_AbstractConfigContainer):

    rrcscale:tuple[float,float]
    rrcratio:tuple[float,float]
    interpolation_modes:Sequence[str]
    hflip:bool
    vflip:bool
    aug3:bool
    randaug:str
    cutmix:bool
    mixup:bool


@dataclass
class _SchConfigContainer(_AbstractConfigContainer):

    lr_name:str
    wd_name:str


@dataclass
class _ModConfigContainer(_AbstractConfigContainer):

    name:str


@dataclass
class _DatConfigContainer(_AbstractConfigContainer):

    name:str
    loc:str
    aug:_AugConfigContainer


@dataclass
class _OptConfigContainer(_AbstractConfigContainer):

    name:str
    learning_rate:float
    weight_decay:float
    opt_epsilon:float
    gradclip:float
    accumulation_steps:int
    amsgrad:bool
    sch:_SchConfigContainer


@dataclass
class _LogConfigContainer(_AbstractConfigContainer):

    logfreq:int
    savedir:str
    debugstdout:bool


@dataclass 
class RunConfig:

    epochs:int
    batch_size:int
    use_devices:Sequence[int]
    targetindices:Sequence[int]
    data_maxlen:int
    mod:_ModConfigContainer
    dat:_DatConfigContainer
    opt:_OptConfigContainer
    log:_LogConfigContainer

    @property
    def world_size(self) -> int:
        return len(self.use_devices)

    @property
    def main_device(self) -> int:
        return self.use_devices[0]
    
    @property
    def use_ddp(self) -> bool:
        return len(self.use_devices) > 1
    

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

    __default_kwargs = {
        'mod': {},
        'dat': {},
        'opt': {},
        'log': {},
        'aug': {},
        'sch': {},
    }

    def __init__(
        self, 
        runworker:Type[AbstractRunWorker],
        use_devices:Sequence[int],
        epochs:int, 
        batch_size:int,
        modelname:str, 
        dataname:str,
        dataloc:str,
        optname:str,
        targetindices:Sequence[int]=[-1],
        learning_rate:float=3e-3,
        weight_decay:float=2e-2,
        opt_epsilon:float=1e-7,
        gradclip:float=1.0,
        accumulation_steps:int=1,
        amsgrad:bool=False,
        logfreq:int=50,
        savedir:str='./runlogs',
        debugstdout:bool=False,
        rrcscale:tuple[float,float]=(.08, 1.0),
        rrcratio:tuple[float,float]=(.75, 4/3),
        interpolation_modes:Sequence[str]=['all'],
        hflip:bool=False,
        vflip:bool=False,
        aug3:bool=False,
        randaug:str='none',
        cutmix:bool=False,
        mixup:bool=False,
        lrsched:str='none',
        wdsched:str='none',
        **_kwargs
    ):
        '''Parses and stores the parameters of a run.

        NOTE: Most of the arguments are based on standard frameworks but can be extended by
              adding keyword arguments in subdictionaries `mod, dat, aug, opt, sch, log` for
              parsing model, data, optimizer, scheduler, and logging for the run.

        Args:
            use_devices (Sequence[int]): Sequence of device indices.
            epochs (int): Number of epochs for run.
            batch_size (int): Batch size per iteration.
            modelname (int): Name of model, parsed using `parse_model`.
            dataname (str): Dataset name, parsed using `parse_data`.
            dataloc (str): Dataset location.
            optname (str): Optimizer name, parsed using `parse_opt`.
            targetindices (Sequence[int]): Indices from dataset that are to be used as targets.
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for optimizer.
            opt_epsilon (float): Epsilon to use for optimizers with momentum est.
            gradclip (float): Gradient clipping, defaults to 1.0.
            accumulation_steps (int): Number of accumulation steps for each batch.
            amsgrad (bool): Flag for using AMSGrad in optimizer.
            logfreq (int): The logging interval in iterations.
            savedir (str): Directory for checkpoints, logs, etc. Defaults to `./runlogs/`.
            debugstdout (bool): Flag for enabling logging to stdout during training.
            rrcscale (tuple[float, float]): Random resize crop scaling.
            rrcratio (tuple[float, float]):  Random resize crop aspect ratio.
            interpolation_modes (Sequence[str]): List of interpolation modes to use.
            hflip (bool): Flag for enabling hflip.
            vflip (bool): Flag for enabling vflip.
            aug3 (bool): Flag for enabling 3augment from DEiTv3.
            randaug (str): Randaug level, defaults to `none`.
            cutmix (bool): Flag for enabling cutmix.
            mixup (bool): Flag for enabling mixup.
            lrsched (str): Type of learning rate scheduler to use.
            lrsched (str): Type of weight decay scheduler to use.
        '''
        self.RunWorker:Type[AbstractRunWorker] = runworker
        kwargs = {**self.__default_kwargs, **_kwargs}

        # Initialize parameter container
        self.cfg:RunConfig = RunConfig(
            epochs,
            batch_size, 
            use_devices, 
            targetindices,
            self._getmaxlen(targetindices),
            mod=_ModConfigContainer(
                modelname
            ),
            dat=_DatConfigContainer(
                dataname, dataloc, _AugConfigContainer(
                    rrcscale, rrcratio, interpolation_modes, hflip, vflip,
                    aug3, randaug, cutmix, mixup
                )
            ),
            opt=_OptConfigContainer(
                optname, learning_rate, weight_decay, opt_epsilon, gradclip, 
                accumulation_steps, amsgrad, _SchConfigContainer(
                    lrsched, wdsched
                )
            ),
            log=_LogConfigContainer(
                logfreq, savedir, debugstdout
            ),
        )

        # Add non-default parameters / configs
        self.cfg.mod.add_attr(**kwargs['mod'])
        self.cfg.dat.add_attr(**kwargs['dat'])
        self.cfg.dat.aug.add_attr(**kwargs['aug'])
        self.cfg.opt.add_attr(**kwargs['opt'])
        self.cfg.opt.sch.add_attr(**kwargs['sch'])
        self.cfg.log.add_attr(**kwargs['log'])

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

