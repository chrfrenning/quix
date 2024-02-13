from setuptools import setup

setup(
    name = 'quix',
    version = '0.1.18',
    description = 'QUIck eXperiment',
    licence = 'GNUv3',
    packages = ['quix', 'quix.run', 'quix.cfg', 'quix.data'],
    install_requires = [
        'toml >= 0.10.2',
        'numpydoc >= 1.6.0',
        'PyYaml >= 6.0',
        'torch >= 2.1.0',
        'torchvision >= 0.16.0',
        'scipy >= 1.10.1'
    ]
)