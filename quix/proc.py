import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler

from contextlib import nullcontext
from typing import Sequence, ContextManager, Optional, Union

from .log import BaseLogHandler

TensorSequence = Union[Tensor, Sequence[Tensor]]

class _AbstractBatchProcessor:

    '''AbstractBatchProcessor defines interface for BatchProcessor.
    '''

    def __init__(self):
        raise NotImplementedError()

    def optimize(
        self, 
        epoch:int,
        iteration:int,
        loss:Optional[Tensor], 
        inputs:Optional[TensorSequence]=None, 
        outputs:Optional[TensorSequence]=None,
        targets:Optional[TensorSequence]=None,
        final_batch:bool=False,
        context:ContextManager=nullcontext(),
        **logging_kwargs
    ) -> None:
        raise NotImplementedError()

class _BatchProcessingContext(ContextManager):

    def __init__(
        self, 
        parent:_AbstractBatchProcessor, 
        epoch:int,
        iteration:int,
        inputs:Optional[TensorSequence]=None, 
        targets:Optional[TensorSequence]=None,
        final_batch:bool=False,
        context:ContextManager=nullcontext(),
        training:bool=True,
        **logging_kwargs
    ):
        self.parent = parent
        self.epoch = epoch
        self.iteration = iteration
        self.inputs = inputs
        self.targets = targets
        self.final_batch = final_batch
        self.context = context
        self.logging_kwargs = logging_kwargs
        self.outputs:Optional[Sequence[Tensor]] = None
        self.loss:Optional[Tensor] = None

    def __enter__(self):
        # TODO: Any setup tasks before entering the block?
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Do optimization and logging with parent
        if exc_type is None:  # If no exceptions, proceed with logging
            self.parent.optimize(
                self.epoch, self.iteration, self.loss, self.inputs, self.outputs, 
                self.targets, self.final_batch, self.context, **self.logging_kwargs
            )

class BatchProcessor(_AbstractBatchProcessor):

    '''Handles the optimization process for batches during training.

    This class manages the optimization process for each batch in the training loop,
    including backward propagation, gradient clipping, optimizer steps, and learning
    rate scheduling. It supports mixed precision training with gradient scaling, 
    gradient accumulation for larger effective batch sizes, and customizable logging.

    Attributes:
        optimizer (Optimizer): The optimizer used for updating model parameters.
        scheduler (Optional[LRScheduler]): Learning rate scheduler.
        scaler (Optional[GradScaler]): Gradient scaler for mixed precision training.
        accumulation_steps (int): Number of steps to accumulate gradients over.
        gradient_clipping (Optional[float]): Threshold for gradient clipping.
        logger (Optional[BaseLogHandler]): Logger for recording training metrics.
        consistent_batch_size (bool): If True, normalizes batch sizes as much as possible.
    
    Methods:
        backward(loss: Tensor): Performs the backward pass on the given loss.
        clip_gradients(): Applies gradient clipping if set.
        optimizer_step(): Performs an optimizer step and checks if it's skipped.
        scheduler_step(step_skipped): Steps the scheduler if the optimizer step wasn't skipped.
        __call__(epoch, iteration, loss, inputs, outputs, targets, final_batch, context, **logging_kwargs):
            Executes the optimization steps for the current batch and logs the results.
    '''    

    def __init__(
        self,
        optimizer:Optimizer,
        scheduler:Optional[LRScheduler]=None,
        scaler:Optional[GradScaler]=None,
        accumulation_steps:int=1,
        gradient_clipping:Optional[float]=None,
        logger:Optional[BaseLogHandler]=None,
        consistent_batch_size:bool=True,
    ):
        '''Initializes the BatchProcessor with the specified parameters.

        Args:
            optimizer (Optimizer): The optimizer used for updating model parameters.
            scheduler (Optional[LRScheduler]): Learning rate scheduler.
            scaler (Optional[GradScaler]): Gradient scaler for mixed precision training.
            accumulation_steps (int): Number of steps to accumulate gradients over.
            gradient_clipping (Optional[float]): Threshold for gradient clipping.
            logger (Optional[BaseLogHandler]): Logger for recording training metrics.
            consistent_batch_size (bool): If True, normalizes batch sizes as much as possible.
        '''
        self.optimizer = optimizer
        self._scheduler = scheduler
        self._logger = logger
        self._scaler = scaler
        self.gradient_clipping = gradient_clipping
        self.accumulation_steps = accumulation_steps
        self.consistent_batch_size = consistent_batch_size
        self._internal_iteration_counter = 0

    @property
    def _opt_params(self):
        # Retrieves optimizer parameters for gradient clipping
        return [p for group in self.optimizer.param_groups for p in group['params']]
    
    def backward(self, loss:Tensor) -> None:
        '''Performs the backward pass for the given loss.

        Args:
            loss (Tensor): The loss tensor to perform backward pass on.
        '''
        if self._scaler is None:
            loss.backward()
            return
        self._scaler.scale(loss).backward() # type: ignore
    
    def clip_gradients(self) -> None:
        '''Applies gradient clipping to the optimizer parameters if specified.
        '''
        if self.gradient_clipping is None:
            return
        if self._scaler is not None:
            self._scaler.unscale_(self.optimizer)
            clip_grad_norm_(self._opt_params, self.gradient_clipping)

    def optimizer_step(self) -> bool:
        '''Performs an optimizer step and checks if it's skipped.

        Returns:
            bool: True if the optimizer step was skipped, otherwise False.
        '''
        step_skipped = False
        if self._scaler is not None:
            # Check whether scaler skipped (can happen with AMP)
            old_scale = self._scaler.get_scale()
            self._scaler.step(self.optimizer)
            self._scaler.update()
            if old_scale < self._scaler.get_scale():
                step_skipped = True
        else:
            self.optimizer.step()

        return step_skipped

    def scheduler_step(self, step_skipped) -> Optional[float]:
        '''Performs a scheduler step if the optimizer step wasn't skipped.

        Args:
            step_skipped (bool): Indicates whether the optimizer step was skipped.

        Returns:
            Optional[float]: The current learning rate after stepping the scheduler, or None.
        '''
        if self._scheduler is not None and not step_skipped:
            cur_lr = self._scheduler.get_last_lr()[0]
            self._scheduler.step()
            return cur_lr
        return .0

    def _perform_optimization(self, iteration, final_batch):
        # Determines whether to perform optimization based on accumulation steps

        if self.consistent_batch_size: 
            # Use internal state, ignoring final_batch
            if self._internal_iteration_counter >= self.accumulation_steps:
                return True
            return False
        
        # Use iteration from training loop
        if (iteration + 1) % self.accumulation_steps == 0 or final_batch:
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
        inputs:Optional[TensorSequence]=None, 
        outputs:Optional[TensorSequence]=None,
        targets:Optional[TensorSequence]=None,
        final_batch:bool=False,
        context:ContextManager=nullcontext(),
        training:bool=True,
        **logging_kwargs
    ) -> None:
        '''Executes the optimization process for the current batch and logs results.

        Args:
            epoch (int): The current epoch number in the training loop.
            iteration (int): The current iteration number in the training loop.
            loss (Tensor): The loss tensor for the current batch.
            inputs (Optional[TensorSequence]): Input tensors for the current batch.
            outputs (Optional[TensorSequence]): Output tensors for the current batch.
            targets (Optional[TensorSequence]): Target tensors for the current batch.
            final_batch (bool): Indicates whether it's the final batch of the epoch.
            context (ContextManager): A context manager for the optimization process.
            **logging_kwargs: Additional keyword arguments for logging.
        '''
        step_skipped = training
        if training:
            with context:
                self.backward(loss)
                self._internal_iteration_counter += 1

                if self._perform_optimization(iteration, final_batch):
                    self.clip_gradients()
                    step_skipped = self.optimizer_step()
                    self.scheduler_step(step_skipped)
                    self.optimizer.zero_grad(set_to_none=True)
                    self._internal_iteration_counter = 0

        if self._logger is None:
            return
        
        self._logger(
            time=time.time(), epoch=epoch, iteration=iteration, loss=loss.item(), 
            inputs=self._clean(inputs), outputs=self._clean(outputs), targets=self._clean(targets),
            final_batch=final_batch, step_skipped=step_skipped, **logging_kwargs
        )

    def __call__(
        self,
        epoch:int,
        iteration:int,
        inputs:Optional[TensorSequence]=None, 
        targets:Optional[TensorSequence]=None,
        final_batch:bool=False,
        context:ContextManager=nullcontext(),
        training:bool=True,
        **logging_kwargs
    ) -> _BatchProcessingContext:
        '''
        Creates and returns an optimization context for a batch training step.

        Args:
            epoch (int): Current epoch number.
            iteration (int): Current iteration number within the epoch.
            inputs (Optional[TensorSequence]): Inputs for the current batch, used for logging.
            targets (Optional[TensorSequence]): Targets for the current batch, used for logging.
            final_batch (bool): Indicates whether this is the final batch of the epoch, optional.
            context (ContextManager): A context manager for additional context handling, optional.
            **logging_kwargs: Additional keyword arguments for logging purposes.

        Returns:
            _BatchProcessingContext: An instance of the optimization context.

        Example:
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
            self, epoch, iteration, inputs, targets, 
            final_batch, context, training, **logging_kwargs
        )
