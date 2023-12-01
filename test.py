'''Quix Test

This module performs unit tests for Quix.
'''
import unittest
from unittest.mock import patch
from dataclasses import is_dataclass, fields
from typing import get_type_hints
from quix.cfg import (
    RunConfig, ModelConfig, DataConfig, AugmentationConfig, OptimizerConfig,
    SchedulerConfig, LogConfig
)

class TestDocstrings(unittest.TestCase):
    # TODO: Make more sophisticated tests

    def test_dataclass_docstrings(self):
        dcs = [
            RunConfig, ModelConfig, DataConfig, AugmentationConfig, 
            OptimizerConfig, SchedulerConfig, LogConfig
        ]
        for dc in dcs:
            if is_dataclass(dc):
                with self.subTest(dataclass=dc.__name__):
                    self.assertDataclassDocstring(dc)

    def assertDataclassDocstring(self, dataclass):
        docstring = dataclass.__doc__
        self.assertIsNotNone(docstring, f"{dataclass.__name__} missing docstring")

        for field in fields(dataclass):
            self.assertIn(
                field.name, 
                docstring, 
                f"Field '{field.name}' not documented in {dataclass.__name__}"
            )
            # Optionally, check for the presence of field types or descriptions
            field_type = get_type_hints(dataclass)[field.name].__name__
            self.assertIn(
                field_type,
                docstring,
                f"Type of field '{field.name}' not documented in {dataclass.__name__}"
            )

class TestArgparseParsing(unittest.TestCase):

    def test_default_argparse_parsing(self):
        test_args = ['test_model', 'mydataset', '/test/dir']
        parser = RunConfig.get_arg_parser()
        args = parser.parse_args(test_args)
        config = RunConfig.from_arg_namespace(args)
        self.assertEqual(config.model, 'test_model')
        self.assertEqual(config.dat.data, 'mydataset')
        self.assertEqual(config.dat.data_path, '/test/dir')
        # TODO: Add more assertions for other fields

    def test_missing_required_args(self):
        # Test behavior when required args are missing
        test_args = ['test_model']  # Omit some required args
        with patch('sys.stderr'), patch('sys.stdout'):
            with self.assertRaises(SystemExit):  # Argparse exits the program on error
                parser = RunConfig.get_arg_parser()
                parser.parse_args(test_args)    
        # TODO: Add more assertions

if __name__ == '__main__':
    # Runs unit tests
    unittest.main()