import torch
from typing import Union

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    
    Updates are performed by
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    """

    def __init__(self, model, decay:float, device:Union[str, torch.device]="cpu"):
        '''Initializes an EMA model instance.

        Parameters
        ----------
        model : nn.Module
            Model to perform averaging over
        decay : float
            Decay value
        device : Union[str, torch.device]
            Device to place EMA model on.
        '''
        self.decay = decay
        super().__init__(model, device, self.ema_avg, use_buffers=True) # type:ignore

    @torch.no_grad
    def ema_avg(self, avg_model_param, model_param, num_averaged):
        return self.decay * avg_model_param + (1 - self.decay) * model_param

