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
configuration classes. 

Submodules
----------
`cfgutils.py`
    Contains utilities for the cfg module.
`config.py`
    Contains main config classes and tools.

Author
------
Marius Aasan <mariuaas@ifi.uio.no>

Contributors
------------
Please consider contributing to the project.
'''
from .config import (
    RunConfig, ModelConfig, OptimizerConfig, DataConfig, LogConfig, 
    add_argument, TDat, TLog, TMod, TOpt
)

# Type alias for RunConfig without generics
StdRunConfig = RunConfig[
    ModelConfig,DataConfig,OptimizerConfig,LogConfig
]