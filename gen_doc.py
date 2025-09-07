import inspect
import textwrap
import re
import itertools
import numbers
import importlib
import sys
import functools
from pathlib import Path

from utils3d._helpers import suppress_traceback


def get_simple_signature(func):
    sig = inspect.signature(func)
    params = []
    for name, param in sig.parameters.items():
        params.append(name)
        # if param.default is param.empty:
        #     params.append(name)
        # else:
        #     params.append(f"{name}={param.default}")
    return f"{func.__name__}({', '.join(params)})"


def get_signature(fn):
    signature = inspect.signature(fn)
    
    signature_str = str(signature)

    signature_str = re.sub(r"<class '.*'>", lambda m: m.group(0).split('\'')[1], signature_str)
    signature_str = re.sub(r"(?<!\.)numpy\.", "numpy_.", signature_str)
    signature_str = re.sub(r"(?<!\.)torch\.", "torch_.", signature_str)
    
    return signature_str

def get_function_source_location(fn):
    fn = inspect.unwrap(fn)
    filepath = Path(inspect.getfile(fn)).relative_to(Path(__file__).parent)
    start_line = inspect.getsourcelines(fn)[1]
    return filepath, start_line


def get_description(fn):
    return fn.__doc__.strip().split("\n")[0]


def collapse_long_text(text):
    if len(text) > 40:
        text = text[:37] + '...'
    return text

if __name__ == "__main__":
    import utils3d.numpy, utils3d.torch
    numpy_impl = utils3d.numpy
    torch_impl = utils3d.torch

    modules = ["transforms", "maps", "mesh", "rasterization", "utils"]
    

    with open("_doc.md", "w", encoding="utf-8") as f:
        for module_name in modules:
            f.write(f"### {module_name.capitalize()}\n\n")
            f.write("| Function | Numpy | Pytorch |\n")
            f.write("| ---- | ---- | ---- |\n")
            numpy_funcs = {name: getattr(utils3d.numpy, name) for name in utils3d.numpy.__modules_all__[module_name]}
            torch_funcs = {name: getattr(utils3d.torch, name) for name in utils3d.torch.__modules_all__[module_name]}
            for fname in sorted(set(numpy_funcs) | set(torch_funcs)):
                doc_column_function = f'`utils3d.{fname}`'
                doc_column_description = ""
                if fname in numpy_funcs:
                    fn = numpy_funcs[fname]
                    filepath, start_line = get_function_source_location(fn)
                    signature = get_simple_signature(fn)
                    if fn.__doc__ is not None:
                        doc_column_description = get_description(fn)
                    doc_column_numpy = f'[`utils3d.np.{signature}`]({filepath}#L{start_line})'
                else:
                    doc_column_numpy = "-"

                if fname in torch_funcs:
                    fn = torch_funcs[fname]
                    filepath, start_line = get_function_source_location(fn)
                    signature =  get_simple_signature(fn)
                    if doc_column_description is None and fn.__doc__ is not None:
                        doc_column_description = get_description(fn)
                    doc_column_torch = f'[`utils3d.pt.{signature}`]({filepath}#L{start_line})'
                else:
                    doc_column_torch = "-"

                f.write(f"| {doc_column_function}<br>{doc_column_description} | {doc_column_numpy} | {doc_column_torch} | \n")
            f.write("\n\n")
