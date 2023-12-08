'''
Quix Run
========

This module provides the `Run` class, a central component of the Quix framework, designed 
to orchestrate the training and validation processes for machine learning models. It handles 
the setup and execution of training runs, leveraging the capabilities of PyTorch and the 
Quix framework's custom functionalities.

The `Run` class integrates various aspects of a typical machine learning pipeline, including 
data loading, model initialization, optimization, and distributed training setup. It offers 
flexibility and convenience for users looking to train models with complex configurations.

Class
-----
Run
    Manages the entire lifecycle of a training and validation run within the Quix framework.

Methods
-------
__init__(cfg: RunConfig)
    Initializes the `Run` instance with a parsed RunConfig.
parse_augmentations()
    Parses and sets up augmentations for the training and validation datasets.
parse_data()
    Prepares data loaders for training and validation datasets.
parse_model()
    Initializes and configures the model based on the provided configuration.
parse_optimization(model: nn.Module, n_samples: Optional[int])
    Sets up the loss function, optimizer, learning rate scheduler, and other optimization-related aspects.
parse_run()
    Parses and prepares all components required for a training/validation run.
process_epoch(...)
    Processes a single epoch of training or validation, handling data loading, model updates, and logging.
run()
    Executes the training and validation process across all epochs.
argparse(...)
    Class method to parse a `Run` instance from dataclasses.
__repr__()
    Returns a string representation of the `Run` instance, useful for debugging and logging.

Example Usage
-------------
Setting up and executing a training run:

```python
from quix_framework import Run, StdRunConfig, ModelConfig, DataConfig, OptimizerConfig, LogConfig

# Parse configuration for the run
cfg = Run.argparse(
    mod=ModelConfig(...),
    dat=DataConfig(...),
    opt=OptimizerConfig(...),
    log=LogConfig(...)
)

# Initialize and run the training process
train_run = Run(cfg)
train_run.run()
```

This module is essential for users of the Quix framework, providing an accessible and streamlined 
interface for conducting training runs with various configurations and requirements.

Author
------
Marius Aasan <mariuaas@ifi.uio.no>
'''
from .runner import Runner