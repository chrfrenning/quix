from __future__ import annotations
import inspect
import warnings
import os
from argparse import ArgumentTypeError
from numpydoc.docscrape import NumpyDocString
from dataclasses import fields, is_dataclass
from typing import (
    Type, Any, Dict, Union, Optional, Callable, TypeVar,
    Sequence, get_args, get_origin
)

T = TypeVar('T')

def _fromenv(envkey:str, typecast:Callable[[str], T]) -> Optional[T]:
    '''Helper method for setting defaults from environment variable.
    '''
    val = os.environ.get(envkey, None)
    return typecast(val) if val else None


def _unified_parser(value:str, types:Sequence[Type], base_types:Sequence[Type], descr:str):
    '''Helper method to parse Union / Base types.
    '''
    for t in types:
        if t in base_types:
            try:
                return t(value)
            except ValueError:
                continue
        elif get_origin(t) is Union:
            return _unified_parser(value, get_args(t), base_types, descr)
    raise ArgumentTypeError(f"Value '{value}' is not valid for {descr}")


def _get_parser(types:Sequence[Type], base_types:Sequence[Type], descr:str):
    '''Factory method for constructing unified parsers.
    '''
    def _parser(x):
        return _unified_parser(x, types, base_types, descr)
    return _parser


def _deffac(val:Any) -> Any:
    return val

def _repr_helper(obj:Any, indent:int=0) -> str:
    '''Helper method for __repr__ to handle nested dataclasses.
    
    Prints in a style similar to PyTorch module printing.
    '''
    pad = ' ' * indent
    repr_str = obj.__class__.__name__ + '(\n'
    
    for fld in fields(obj):
        value = getattr(obj, fld.name)
        if is_dataclass(value):
            value_str = _repr_helper(value, indent + 2)
        else:
            value_str = repr(value)
        repr_str += f"{pad}  {fld.name}: {value_str}\n"

    repr_str += pad + ')'
    return repr_str


def _parse_docstring(docstring:str) -> Dict[str, Any]:
    '''Helper method to parse docstrings.
    '''
    doc = NumpyDocString(docstring)
    return {item.name: {'help': '\n'.join(item.desc)} for item in doc['Attributes']}


def _extract_metadata(cls) -> Dict[str, Any]:
    '''Helper function to extract docstring metadata.
    '''
    metadata = {}
    for c in reversed(cls.__mro__):
        cls_docstring = inspect.getdoc(c)
        if cls_docstring:
            metadata.update(_parse_docstring(cls_docstring))
    return metadata


def metadata_decorator(cls):
    '''Metadata decorator.

    This function acts on dataclasses to embed docstring
    descriptions into field metadata. This is later passed to the
    arparser for parsing a config file.

    Parameters
    ----------
    cls
        A class instance for the metatype decorator
    '''
    metadata_dict = _extract_metadata(cls)
    dataclass_fields = {f.name for f in fields(cls)}
    docstring_attr = set(metadata_dict.keys())
    missing_in_docstring = dataclass_fields - docstring_attr
    missing_in_fields = docstring_attr - dataclass_fields

    if missing_in_docstring:
        warnings.warn(
            f"Warning: Fields missing in docstring for {cls.__name__}: {missing_in_docstring}"
        )
    if missing_in_fields:
        warnings.warn(
            f"Warning: Docstring attributes not found as fields in {cls.__name__}: {missing_in_fields}"
        )

    for f in fields(cls):
        if f.name in metadata_dict:
            f.metadata = {'default': f.default, **metadata_dict[f.name], **f.metadata} # type: ignore

    return cls