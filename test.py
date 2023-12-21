'''Quix Test

This module performs unit tests for Quix.
'''
import unittest
import os
from unittest.mock import patch
from dataclasses import is_dataclass, fields
from typing import get_type_hints
from quix.cfg import (
    RunConfig, ModelConfig, DataConfig, OptimizerConfig,
    LogConfig, add_argument
)

# class TestDocstrings(unittest.TestCase):
#     # TODO: Make more sophisticated tests

#     def test_dataclass_docstrings(self):
#         dcs = [
#             RunConfig, ModelConfig, DataConfig, AugmentationConfig, 
#             OptimizerConfig, SchedulerConfig, LogConfig
#         ]
#         for dc in dcs:
#             if is_dataclass(dc):
#                 with self.subTest(dataclass=dc.__name__):
#                     self.assertDataclassDocstring(dc)

#     def assertDataclassDocstring(self, dataclass):
#         docstring = dataclass.__doc__
#         self.assertIsNotNone(docstring, f"{dataclass.__name__} missing docstring")

#         for field in fields(dataclass):
#             self.assertIn(
#                 field.name, 
#                 docstring, 
#                 f"Field '{field.name}' not documented in {dataclass.__name__}"
#             )
#             # Optionally, check for the presence of field types or descriptions
#             field_type = get_type_hints(dataclass)[field.name].__name__
#             self.assertIn(
#                 field_type,
#                 docstring,
#                 f"Type of field '{field.name}' not documented in {dataclass.__name__}"
#             )

class TestArgparseParsing(unittest.TestCase):

    def test_default_argparse_parsing(self):
        test_args = ['--model', 'test_model', '--dataset', 'mydataset', '--data-path', '/test/dir']
        config = RunConfig.argparse(_testargs=test_args)
        self.assertEqual(config.mod.model, 'test_model')
        self.assertEqual(config.dat.dataset, 'mydataset')
        self.assertEqual(config.dat.data_path, '/test/dir')
        # TODO: Add more assertions for other fields

    def test_missing_required_args(self):
        # Test behavior when required args are missing
        test_args = ['--model', 'test_model']  # Omit some required args
        with patch('sys.stderr'), patch('sys.stdout'):
            with self.assertRaises(SystemExit):  # Argparse exits the program on error
                parser = RunConfig.argparse(_testargs=test_args)

        # TODO: Add more assertions

    def test_json_config(self):
        # Test behavior with json file
        cfgpath = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__), 'rsc', 'test.json')
        )
        test_args = ['--cfgfile', cfgpath]
        config = RunConfig.argparse(_testargs=test_args)
        self.assertEqual(config.mod.model, 'MyModel')
        self.assertEqual(config.dat.dataset, 'MyData')
        self.assertEqual(config.dat.data_path, '/work2/litdata/')
        # TODO: Add more assertions

    def test_yml_config(self):
        # Test behavior with yaml file
        cfgpath = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__), 'rsc', 'test.yml')
        )
        test_args = ['--cfgfile', cfgpath]
        config = RunConfig.argparse(_testargs=test_args)
        self.assertEqual(config.mod.model, 'MyModel')
        self.assertEqual(config.dat.dataset, 'MyData')
        self.assertEqual(config.dat.data_path, '/work2/litdata/')
        # TODO: Add more assertions


    def test_toml_config(self):
        # Test behavior with yaml file
        cfgpath = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__), 'rsc', 'test.toml')
        )
        test_args = ['--cfgfile', cfgpath]
        config = RunConfig.argparse(_testargs=test_args)
        self.assertEqual(config.mod.model, 'MyModel')
        self.assertEqual(config.dat.dataset, 'MyData')
        self.assertEqual(config.dat.data_path, '/work2/litdata/')
        # TODO: Add more assertions

    def test_nested_toml_config(self):
        # Test behavior with yaml file
        cfgpath = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__), 'rsc', 'testnest.toml')
        )
        test_args = ['--cfgfile', cfgpath]
        config = RunConfig.argparse(_testargs=test_args)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.mod.model, 'MyModelNested')
        self.assertEqual(config.dat.dataset, 'MyDataSet')
        self.assertEqual(config.dat.data_path, '/path/to/data')
        self.assertEqual(config.opt.lr, 1e-3)
        self.assertEqual(config.opt.weight_decay, 1e-5)
        self.assertEqual(config.dat.workers, 3)
        self.assertEqual(config.dat.prefetch, 1)

    def test_custom_modelconfig(self):
        class MyModelConfig(ModelConfig):
            '''Testing model config.

            Attributes
            ----------
            milkshake : str
                Takes all the boys to the yard.
            damnright : int
                I might have to charge.
            '''
            milkshake:str = 'Better than yours'
            damnright:int = 100000

        test_args = ['--model', 'test_model', '--dataset', 'mydataset', '--data-path', '/test/dir']
        config = RunConfig.argparse(modcfg=MyModelConfig, _testargs=test_args)
        self.assertEqual(config.mod.milkshake, 'Better than yours')
        yours : int = 0
        self.assertGreater(config.mod.damnright, yours)


if __name__ == '__main__':
    # Runs unit tests
    unittest.main()