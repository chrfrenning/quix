# quix: QUIck eXperiment

<img src="rsc/quix.png" alt="quix in full glory" width="300"/>

**quix** is a PyTorch framework for streamlining experiment design for ML tasks. 

## the TLDR;

building pipelines for large ML experiments for HPC is not only tedious, but boring. **quix** is here to help to make your life **queasier** by being a quick way to speed up your latest ML experiments.
 
- **quix** only requires you to
    - write a model
    - write a parser for your model
    - add some form of training config

you'll be up and running in no time! it's the **quix**$^{\mathrm{TM}}$ guarantee! **quix** will even support your favorite config format; `json`, `yaml` or argparse with command line arguments. hopefully your favorite format is in that list of three things. it's the **quix**$^{\mathrm{TM}}$ way!

of course, **quix** also allows you to extend your framework as you please, as long as you do it in a way **quix** understands!
the goal of **quix** is to abstract away the most common components of your training loop with pretty solid defaults and components so **YOU** can spend time on other things, like buying **quix**$^{\mathrm{TM}}$ merchandise, or even doing research! it is as **quix** and **queasy**$^{\mathrm{TM}}$ as you like!

### some design principles

sure, you could use `lightning` but it does not have full support for ROCm, and it does not offer a way to cohesively do data management. **quix** aims to build a relatively minimal but simple extendable framework with minimal dependencies 
(only Python 3.10, `torch`, `torchvision`, (`scipy` for some encoder formats, and `pyyaml` for config files) are required.

**quix** is build around the following goals
1. *provide a way to parse and extend configurations for experiments that feels natural, and can be extended hierarchically for sub-experiments.*
2. *provide a cohesive method of working with and compiling multimodal vision datasets that is optimized for HPC resources to facilitate quick sharing of data between researchers.*
3. *provide a standard for abstratcing away the heavy lifting in SotA model optimizations*
4. *provide a standard for cohesively parsing and running experiments.*
5. *all this should be done with minimal external dependencies.*
    - *certain current dependencies might be dropped in the future (`scipy` and `pyyaml`)*
6. *submodules of **quix** should be possible to use in isolation, and **quix** yearns to be as modular as possible.*
7. **quix** *really wants to use good type hinting for making sure your experiments run as you expect each time*.

essentially, when **quix** grows up someday, it wants to be a compact form of `lightning`, with less bells and whistles and data management build in.

#### maybe even more important

we omitted to mention a certain goal of **quix**; namely to provide a cohesive way of packaging models and experiments for reproducibility. we eventually want to package data and model checkpoints in a database such that we can build on each others experiences and modelling efforts without reinventing the wheel each time!


## more speciquix

the **quix** framework comprises several modules, each serving a distinct purpose in data handling, processing, and model training. here's a brief summary of the current modules:

1. **dataset** (`data`): 
    - manages datasets stored in local tar shards, optimizing for local filesystems with high-speed access.
    - `QuixDataset` simplifies dataset handling, supporting shuffling and batching.
    - `QuixUnionDataset` allows for concatenating multiple `QuixDataset` instances, enabling simultaneous sampling from all datasets.
    - **data submodules** (`encoders`, `writer`):
        - `encoders` provides a variety of data encoder and decoder classes (`CLS`, `PIL`, `NPY`, `RLE`, `SEG8`, `SEG16`, `SEG24`, `SEG32`) designed to handle a variety of data modalities. 
        - `writer` facilitates efficient creation and management of large datasets in sharded tar format.
            - `QuixWriter` is a high-level utility for creating and managing datasets, handling storage patterns, data writing, and metadata management.


2. **configuration** (`cfg`):
    - offer a structured approach to defining configuration parameters for training runs.
    - allows parsing and defining modules using python `dataclasses` with built in support for `argparse` functionality.
    - can also parse run configurations from nested `json` and `yaml` files for structured subexperiments.
    - `RunConfig` is an extendable configuration manager.
        - the user can define custom subclasses (currently `ModelConfig`, `DataConfig`, `OptimizerConfig`, `LogConfig`) which are passed to `RunConfig` which handles all parsing.

3. **processor** (`proc`):
    - includes a powerful set of batch optimization routines for handling optimizers, scheduling, scaling with amp, model averaging, gradient accumulation, and logging.
    - `BatchProcessor` handles this by providing a `ContextManager` and can be used independent of the rest of the **quix** framework.
    - essentially, given your configuration you launch a context using 
    ```python
    for ep in range(epochs):
        for it, (inputs, targets) in enumerate(dataloader):
            with processor(ep, it, opt, sch, inputs=inputs, targets=targets) as proc:
                proc.outputs = model(inputs)
                proc.loss = loss_fn(inputs, proc.outputs)
            # When exiting the context, all optimization is handled
            # by the BatchProcessor automagically
    ```

3. **runner**:
    - the `Runner` class orchestrates the entire lifecycle of training and validation runs, integrating data loading, model setup, and optimization processes, including multi-node or multi-gpu setups with DDP using `torch.distributed.run` or `torchrun`.
    - all aspects of `Runner` can be customized by subclassing with methods:
        - `parse_augmentations` for parsing your specific augmentation setup.
        - `parse_data` for parsing custom datasets directly.
        - `parse_model` for customizing the model initialization from your configuration.
        - `parse_scheduler` for setting a custom learning rate scheduler.
        - `parse_run` for completely determining the parsing of the run from scratch.
        - `run` for customizing the full run process as you like...
        - ...as well as alternative methods scaling, model averaging, checkpointing, custom DDP setup, and logging.

## very speciquix (docs)

a full specification of the **speciquix** will be addressed in the docs.

currently, you can learn all about **quix** by looking really hard at the source code for a while. feel free to read the docstrings of some functions to understand more about **quix**!!! **quix** reminds you to look away if you feel queasy!


## contribute?

please consider helping **quix**$^\mathrm{TM}$ become a fully fledged adult by contributing to it's ongoing development. you can do this by testing quix and contributing to the codebase.
