import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp.grad_scaler import GradScaler

from contextlib import nullcontext
from typing import Sequence, Callable, ContextManager, Optional, Union

from .log import LogCollator

TensorSequence = Union[Tensor, Sequence[Tensor]]
CallableContext = Callable[[], ContextManager]

class _AbstractBatchProcessor:

    def __init__(self):
        raise NotImplementedError()

    def optimize(
        self, 
        epoch:int,
        iteration:int,
        loss:Optional[Tensor], 
        optimizer:Optimizer,
        scheduler:Optional[LRScheduler]=None,
        scaler:Optional[GradScaler]=None,
        model:Optional[nn.Module]=None,
        averaged_model:Optional[AveragedModel]=None,
        inputs:Optional[TensorSequence]=None, 
        outputs:Optional[TensorSequence]=None,
        targets:Optional[TensorSequence]=None,
        final_batch:bool=False,
        context:CallableContext=nullcontext,
        training:bool=False,
        **logging_kwargs
    ) -> None:
        raise NotImplementedError()

class _BatchProcessingContext(ContextManager):

    def __init__(
        self, 
        parent:_AbstractBatchProcessor, 
        epoch:int,
        iteration:int,
        optimizer:Optimizer,
        scheduler:Optional[LRScheduler]=None,
        scaler:Optional[GradScaler]=None,
        model:Optional[nn.Module]=None,
        averaged_model:Optional[AveragedModel]=None,
        inputs:Optional[TensorSequence]=None, 
        targets:Optional[TensorSequence]=None,
        final_batch:bool=False,
        context:CallableContext=nullcontext,
        training:bool=True,
        **logging_kwargs
    ):
        self.parent = parent
        self.epoch = epoch
        self.iteration = iteration
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.model = model
        self.averaged_model = averaged_model
        self.inputs = inputs
        self.targets = targets
        self.final_batch = final_batch
        self.context = context
        self.training = training
        self.logging_kwargs = logging_kwargs
        self.outputs:Optional[TensorSequence] = None
        self.loss:Optional[Tensor] = None

    def __enter__(self):
        # TODO: Any setup tasks before entering the block?
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Do optimization and logging with parent
        if exc_type is None:  # If no exceptions, proceed with logging
            self.parent.optimize(
                self.epoch, self.iteration, self.loss, self.optimizer, 
                self.scheduler, self.scaler, self.model, self.averaged_model, 
                self.inputs, self.outputs, self.targets, self.final_batch, 
                self.context, self.training, **self.logging_kwargs
            )

class BatchProcessor(_AbstractBatchProcessor):

    '''Handles the optimization process for batches during training.

    This class manages the optimization process for each batch in the training loop,
    including backward propagation, gradient clipping, optimizer steps, and learning
    rate scheduling. It supports mixed precision training with gradient scaling, 
    gradient accumulation for larger effective batch sizes, and customizable logging.

    Attributes
    ----------
    accumulation_steps : int
        Number of steps to accumulate gradients over.
    gradient_clipping : Optional[float]
        Threshold for gradient clipping.
    logger : Optional[BaseLogHandler]
        Logger for recording training metrics.
    consistent_batch_size : bool
        If True, normalizes batch sizes as much as possible.
    
    Methods
    -------
    backward(loss: Tensor)
        Performs the backward pass on the given loss.
    clip_gradients()
        Applies gradient clipping if set.
    optimizer_step()
        Performs an optimizer step and checks if it's skipped.
    scheduler_step(step_skipped)
        Steps the scheduler if the optimizer step wasn't skipped.
    __call__(epoch, iteration, loss, inputs, outputs, targets, final_batch, context, **logging_kwargs)
        Executes the optimization steps for the current batch and logs the results.
    '''

    def __init__(
        self,
        accumulation_steps:int=1,
        average_steps:int=0,
        average_warmup_epochs:int=0,
        gradient_clipping:Optional[float]=None,
        logger:Optional[LogCollator]=None,
        consistent_batch_size:bool=True,
    ):
        '''Initializes the BatchProcessor with the specified parameters.

        Parameters
        ----------
        accumulation_steps : int
            Number of steps to accumulate gradients over.
        average_steps : int
            Number of steps between average updates.
        average_warmup_epochs : int
            Number of warmup epochs before applying model averaging.
        gradient_clipping : Optional[float]
            Threshold for gradient clipping.
        logger : Optional[BaseLogHandler]
            Logger for recording training metrics.
        consistent_batch_size : bool
            If True, normalizes batch sizes as much as possible.
        '''
        self._logger = logger
        self.gradient_clipping = gradient_clipping
        self.accumulation_steps = accumulation_steps
        self.average_steps = average_steps
        self.average_warmup_epochs = average_warmup_epochs
        self.consistent_batch_size = consistent_batch_size
        self._acc_iteration = 0
        self._avg_update = average_steps * accumulation_steps
        self._avg_iteration = 0
        self._skip_averaging = self.average_steps > 0

    @staticmethod
    def _opt_params(optimizer:Optimizer):
        # Retrieves optimizer parameters for gradient clipping
        return [p for group in optimizer.param_groups for p in group['params']]
    
    def backward(self, loss:Tensor, scaler:Optional[GradScaler]=None) -> None:
        '''Performs the backward pass for the given loss.

        Parameters
        ----------
        loss : Tensor
            The loss tensor to perform backward pass on.
        scaler : GradientScaler
            An optional gradient scaler instance.
        '''
        if scaler is None:
            loss.backward()
            return
        scaler.scale(loss).backward() # type: ignore
    
    def clip_gradients(self, optimizer:Optimizer, scaler:Optional[GradScaler]=None) -> None:
        '''Applies gradient clipping to the optimizer parameters if specified.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer whose parameters we are adding gradient clipping to.
        scaler : GradScaler
            An optional GradScaler instance.
        '''
        if self.gradient_clipping is None:
            return
        if scaler:
            scaler.unscale_(optimizer)
            clip_grad_norm_(self._opt_params(optimizer), self.gradient_clipping)

    def optimizer_step(self, optimizer:Optimizer, scaler:Optional[GradScaler]=None) -> bool:
        '''Performs an optimizer step and checks if it's skipped.

        When using Automatic Mixed Precision (AMP), a step is skipped
        if it contains NaNs. This throws a warning, but can be avoided
        by explicitly checking for skips. The check also allows us to
        determine consistent batch sizes with AMP.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer for which we want to perform a gradient step.
        scaler : GradScaler
            An optional GradScaler instance.

        Returns
        -------
        bool
            True if the optimizer step was skipped, otherwise False.
        '''
        step_skipped = False
        if scaler:
            # Check whether scaler skipped (can happen with AMP)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if old_scale < scaler.get_scale():
                step_skipped = True
        else:
            optimizer.step()

        return step_skipped

    def scheduler_step(self, step_skipped:bool, scheduler:Optional[LRScheduler]=None) -> Optional[float]:
        '''Performs a scheduler step if the optimizer step wasn't skipped.

        Parameters
        ----------
        step_skipped : bool
            Indicates whether the optimizer step was skipped.

        Returns
        -------
        Optional[float]
            The current learning rate after stepping the scheduler, or None.
        '''
        if scheduler and not step_skipped:
            cur_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            return cur_lr
        return .0

    def _perform_optimization(self, iteration, final_batch):
        # Determines whether to perform optimization based on accumulation steps

        if self.consistent_batch_size: 
            # Use internal state, ignoring final_batch
            if self._acc_iteration >= self.accumulation_steps:
                return True
            return False
        
        # Use iteration from training loop
        if (iteration + 1) % self.accumulation_steps == 0 or final_batch:
            return True
        return False
    
    def _perform_average_update(self, iteration, final_batch):
        # Determines whether to perform average update
        if self._skip_averaging:
            return False
        if self.consistent_batch_size: 
            # Use internal state, ignoring final_batch
            if self._avg_iteration >= self.average_steps:
                return True
            return False
        
        # Use iteration from training loop
        if (iteration + 1) % self._avg_update == 0 or final_batch:
            return True
        return False
    
    def _clean(self, tseq:Optional[TensorSequence]) -> Optional[TensorSequence]:
        # Cleans tensors for logging
        if torch.is_tensor(tseq) and isinstance(tseq, Tensor):
            return tseq.detach().cpu()
        if isinstance(tseq, Sequence):
            return [t.detach().cpu() for t in tseq]
        return tseq
        
    def optimize(
        self, 
        epoch:int,
        iteration:int,
        loss:Tensor, 
        optimizer:Optimizer,
        scheduler:Optional[LRScheduler]=None,
        scaler:Optional[GradScaler]=None,
        model:Optional[nn.Module]=None,
        averaged_model:Optional[AveragedModel]=None,
        inputs:Optional[TensorSequence]=None, 
        outputs:Optional[TensorSequence]=None,
        targets:Optional[TensorSequence]=None,
        final_batch:bool=False,
        context:CallableContext=nullcontext,
        training:bool=True,
        **logging_kwargs
    ) -> None:
        '''Executes the optimization process for the current batch and logs results.

        Parameters
        ----------
        epoch : int
            The current epoch number in the training loop.
        iteration : int
            The current iteration number in the training loop.
        loss : Tensor
            The loss tensor for the current batch.
        optimizer : Optimizer
            Training optimizer. Ignored if training = False.
        scheduler:Optional[LRScheduler]
            An optional scheduler for the batch optimization process.
        scaler:Optional[GradScaler]
            An optional scaler for the batch optimization process.
        model:Optional[nn.Module]
            An optional model for the batch optimization process. Only used in averaged model.
        averaged_model:Optional[AveragedModel]
            An optional averaged model for the batch optimization process.
        inputs : Optional[TensorSequence]
            Input tensors for the current batch.
        outputs : Optional[TensorSequence]
            Output tensors for the current batch.
        targets : Optional[TensorSequence]
            Target tensors for the current batch.
        final_batch : bool
            Indicates whether it's the final batch of the epoch.
        context : ContextManager
            A context manager for the optimization process.
        training : bool
            Boolean whether to apply optimization or just log validation.
        **logging_kwargs
            Additional keyword arguments for logging.
        '''
        step_skipped = training
        if training:
            with context():
                self.backward(loss)
                self._acc_iteration += 1

                if self._perform_optimization(iteration, final_batch):
                    self.clip_gradients(optimizer, scaler)
                    step_skipped = self.optimizer_step(optimizer, scaler)
                    self.scheduler_step(step_skipped, scheduler)
                    optimizer.zero_grad(set_to_none=True)
                    self._acc_iteration = 0
                    self._avg_iteration += 1
                
                if self._perform_average_update(iteration, final_batch):
                    if model and averaged_model:
                        averaged_model.update_parameters(model)
                        if epoch < self.average_warmup_epochs:
                            averaged_model.n_averaged.fill_(0)
                    self._avg_iteration = 0

        if self._logger:
            last_lr = None
            if scheduler is not None:
                last_lr = scheduler.get_last_lr()[0]
            self._logger(
                time=time.time(), epoch=epoch, iteration=iteration, loss=loss.item(), 
                inputs=self._clean(inputs), outputs=self._clean(outputs), targets=self._clean(targets),
                final_batch=final_batch, step_skipped=step_skipped, last_lr=last_lr, training=training,
                **logging_kwargs
            )

    def __call__(
        self, 
        epoch:int,
        iteration:int,
        optimizer:Optimizer,
        scheduler:Optional[LRScheduler]=None,
        scaler:Optional[GradScaler]=None,
        model:Optional[nn.Module]=None,
        averaged_model:Optional[AveragedModel]=None,
        inputs:Optional[TensorSequence]=None, 
        targets:Optional[TensorSequence]=None,
        final_batch:bool=False,
        context:CallableContext=nullcontext,
        training:bool=True,
        **logging_kwargs
    ) -> _BatchProcessingContext:
        '''Creates and returns an optimization context for a batch training step.

        Parameters
        ----------
        epoch : int
            The current epoch number in the training loop.
        iteration : int
            The current iteration number in the training loop.
        optimizer : Optimizer
            Training optimizer. Ignored if training = False.
        scheduler:Optional[LRScheduler]
            An optional scheduler for the batch optimization process.
        scaler:Optional[GradScaler]
            An optional scaler for the batch optimization process.
        model:Optional[nn.Module]
            An optional model for the batch optimization process. Only used in averaged model.
        averaged_model:Optional[AveragedModel]
            An optional averaged model for the batch optimization process.
        inputs : Optional[TensorSequence]
            Input tensors for the current batch.
        targets : Optional[TensorSequence]
            Target tensors for the current batch.
        final_batch : bool
            Indicates whether it's the final batch of the epoch.
        context : ContextManager
            A context manager for the optimization process.
        training : bool
            Boolean whether to apply optimization or just log validation.
        **logging_kwargs
            Additional keyword arguments for logging.

        Returns
        -------
        _BatchProcessingContext
            An instance of the optimization context.

        Example
        -------
        ```
        for epoch in range(num_epochs):
            for iteration, (inputs, targets) in enumerate(dataloader):
                final_batch = iteration == len(dataloader) - 1
                with batch_optimizer(epoch, iteration, inputs, targets, final_batch, context) as opt:
                    opt.outputs = model(inputs)
                    opt.loss = loss_function(o.outputs, targets)
                
                # Optimization and logging is now performed upon exit of context
                ...
        ```
        '''        
        return _BatchProcessingContext(
            self, epoch, iteration, optimizer, scheduler, scaler,
            model, averaged_model, inputs, targets, 
            final_batch, context, training, **logging_kwargs
        )
