# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Miscellaneous utility classes and functions."""
from collections import namedtuple
import time
import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import json
import os
import shutil
import sys
import types
import io
import pickle
import re
# import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid
import boto3
import threading
from contextlib import ContextDecorator
from contextlib import contextmanager, nullcontext

from distutils.util import strtobool
from typing import Any, List, Tuple, Union
import importlib
from loguru import logger
# import wandb
import torch
import psutil
import subprocess

import random
import string
import pdb

# Util classes
# ------------------------------------------------------------------------------------------


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None


# Cache directories
# ------------------------------------------------------------------------------------------

_dnnlib_cache_dir = None


def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path


def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)


# Small util functions
# ------------------------------------------------------------------------------------------


def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def format_time_brief(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m".format(s // (60 * 60), (s // 60) % 60)
    else:
        return "{0}d {1:02}h".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24)


def ask_yes_no(question: str) -> bool:
    """Ask the user the question until the user inputs a valid answer."""
    while True:
        try:
            print("{0} [y/n]".format(question))
            return strtobool(input().lower())
        except ValueError:
            pass


def tuple_product(t: Tuple) -> Any:
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result *= v

    return result


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double
}


def get_dtype_and_ctype(type_obj: Any) -> Tuple[np.dtype, Any]:
    """Given a type name string (or an object having a __name__ attribute), return matching Numpy and ctypes types that have the same size in bytes."""
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    assert type_str in _str_to_ctype.keys()

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    assert my_dtype.itemsize == ctypes.sizeof(my_ctype)

    return my_dtype, my_ctype


def is_pickleable(obj: Any) -> bool:
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------

def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name)  # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    module = obj.__module__
    if module == '__main__':
        module = os.path.splitext(os.path.basename(sys.modules[module].__file__))[0]
    return module + "." + obj.__name__


# File system helpers
# ------------------------------------------------------------------------------------------

def list_dir_recursively_with_ignore(dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False) -> List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


# URL helpers
# ------------------------------------------------------------------------------------------

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)

# ------------------------------------------------------------------------------------------
# util function modified from https://github.com/nv-tlabs/LION/blob/0467d2199076e95a7e88bafd99dcd7d48a04b4a7/utils/model_helper.py
def import_class(model_str):
    from torch_utils.dist_utils import is_rank0
    if is_rank0():
        logger.info('import: {}', model_str)
    p, m = model_str.rsplit('.', 1)
    mod = importlib.import_module(p)
    Model = getattr(mod, m)
    return Model

class ScopedTorchProfiler(ContextDecorator):
    """
    Marks ranges for both nvtx profiling (with nsys) and torch autograd profiler
    """
    __global_counts = {}
    enabled=False

    def __init__(self, unique_name: str):
        """
        Names must be unique!
        """
        ScopedTorchProfiler.__global_counts[unique_name] = 0
        self._name = unique_name
        self._autograd_scope = torch.profiler.record_function(unique_name)
    
    def __enter__(self):
        if ScopedTorchProfiler.enabled:
            torch.cuda.nvtx.range_push(self._name)
        self._autograd_scope.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self._autograd_scope.__exit__(exc_type, exc_value, traceback)
        if ScopedTorchProfiler.enabled:
            torch.cuda.nvtx.range_pop()

class TimingsMonitor():
    CUDATimer = namedtuple('CUDATimer', ['start', 'end'])
    def __init__(self, device, enabled=True, timing_names:List[str]=[], cuda_timing_names:List[str]=[]):
        """
        Usage:
            tmonitor = TimingsMonitor(device)
            for i in range(n_iter):
                # Record arbitrary scopes
                with tmonitor.timing_scope('regular_scope_name'):
                    ...
                    with tmonitor.cuda_timing_scope('nested_scope_name'):
                        ...
                with tmonitor.cuda_timing_scope('cuda_scope_name'):
                    ...
                tmonitor.record_timing('duration_name', end_time - start_time)

                # Gather timings
                tmonitor.record_all_cuda_timings()
                tmonitor.update_all_averages()
                averages = tmonitor.get_average_timings()
                all_timings = tmonitor.get_timings()

        Two types of timers, standard report timing and cuda timings.
        Cuda timing supports scoped context manager cuda_event_scope.
        Args:
            device: device to time on (needed for cuda timers)
            # enabled: HACK to only report timings from rank 0, set enabled=(global_rank==0)
            timing_names: timings to report optional (will auto add new names)
            cuda_timing_names: cuda periods to time optional (will auto add new names)
        """
        self.enabled=enabled
        self.device = device

        # Normal timing
        # self.all_timings_dict = {k:None for k in timing_names + cuda_timing_names}
        self.all_timings_dict = {}
        self.avg_meter_dict = {}

        # Cuda event timers to measure time spent on pushing data to gpu and on training step
        self.cuda_event_timers = {}

        for k in timing_names:
            self.add_new_timing(k)

        for k in cuda_timing_names:
            self.add_new_cuda_timing(k)

        # Running averages
        # self.avg_meter_dict = {k:AverageMeter() for k in self.all_timings_dict}
    
    def add_new_timing(self, name):
        self.avg_meter_dict[name] = AverageMeter()
        self.all_timings_dict[name] = None
    
    def add_new_cuda_timing(self, name):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self.cuda_event_timers[name] = self.CUDATimer(start=start_event, end=end_event)
        self.add_new_timing(name)
    
    def clear_timings(self):
        self.all_timings_dict = {k:None for k in self.all_timings_dict}

    def get_timings(self):
        return self.all_timings_dict

    def get_average_timings(self):
        return {k:v.avg for k,v in self.avg_meter_dict.items()}

    def update_all_averages(self):
        """
        Once per iter, when timings have been finished recording, one should
        call update_average_iter to keep running average of timings.
        """
        for k,v in self.all_timings_dict.items():
            if v is None:
                print("none_timing", k)
                continue
            self.avg_meter_dict[k].update(v)
    
    def record_timing(self, name, value):
        if name not in self.all_timings_dict: self.add_new_timing(name)
        # assert name in self.all_timings_dict
        self.all_timings_dict[name] = value

    def _record_cuda_event_start(self, name):
        if name in self.cuda_event_timers:
            self.cuda_event_timers[name].start.record(
                torch.cuda.current_stream(self.device))

    def _record_cuda_event_end(self, name):
        if name in self.cuda_event_timers:
            self.cuda_event_timers[name].end.record(
                torch.cuda.current_stream(self.device))

    @contextmanager
    def cuda_timing_scope(self, name, profile=True):
        if name not in self.all_timings_dict: self.add_new_cuda_timing(name)
        with ScopedTorchProfiler(name) if profile else nullcontext():
            self._record_cuda_event_start(name)
            try:
                yield
            finally:
                self._record_cuda_event_end(name)

    @contextmanager
    def timing_scope(self, name, profile=True):
        if name not in self.all_timings_dict: self.add_new_timing(name)
        with ScopedTorchProfiler(name) if profile else nullcontext():
            start_time = time.time()
            try:
                yield
            finally:
                self.record_timing(name, time.time()-start_time)
    
    def record_all_cuda_timings(self):
        """ After all the cuda events call this to synchronize and record down the cuda timings. """
        for k, events in self.cuda_event_timers.items():
            with torch.no_grad():
                events.end.synchronize()
                # Convert to seconds
                time_elapsed = events.start.elapsed_time(events.end)/1000.
                self.all_timings_dict[k] = time_elapsed

def init_s3(config_file):
    config = json.load(open(config_file, 'r'))
    s3_client = boto3.client("s3", **config)
    return s3_client

def download_from_s3(file_path, target_path, cfg):
    tic = time.time()
    s3_client = init_s3(cfg.checkpoint.write_s3_config)  # use to test the s3_client can be init
    bucket_name = file_path.split('/')[2]
    file_key = file_path.split(bucket_name+'/')[-1]
    print(bucket_name, file_key)
    s3_client.download_file(bucket_name, file_key, target_path)
    logger.info(f'finish download from ! s3://{bucket_name}/{file_key} to {target_path} %.1f sec'%(
        time.time() - tic))

def upload_to_s3(buffer, bucket_name, key, config_dict):
    logger.info(f'start upload_to_s3! bucket_name={bucket_name}, key={key}')
    tic = time.time()
    s3 = boto3.client('s3', **config_dict)
    s3.put_object(Bucket=bucket_name, Key=key, Body=buffer.getvalue())
    logger.info(f'finish upload_to_s3! s3://{bucket_name}/{key} %.1f sec'%(time.time() - tic))

def write_ckpt_to_s3(cfg, all_model_dict, ckpt_name):
    buffer = io.BytesIO()
    tic = time.time()
    torch.save(all_model_dict, buffer)  # take ~0.25 sec
    # logger.info('write ckpt to buffer: %.2f sec'%(time.time() - tic))
    group, name = cfg.outdir.rstrip("/").split("/")[-2:]
    key = f"checkpoints/{group}/{name}/ckpt/{ckpt_name}"
    bucket_name = cfg.checkpoint.write_s3_bucket
     
    s3_client = init_s3(cfg.checkpoint.write_s3_config)  # use to test the s3_client can be init

    config_dict = json.load(open(cfg.checkpoint.write_s3_config, 'r'))
    upload_thread = threading.Thread(target=upload_to_s3, args=(buffer, bucket_name, key, config_dict))
    upload_thread.start()
    path = f"s3://{bucket_name}/{key}" 
    return path

def upload_file_to_s3(cfg, file_path, key_name=None):
    # file_path is the local file path, can be a yaml file
    # this function is used to upload the ckecpoint only
    tic = time.time()
    group, name = cfg.outdir.rstrip("/").split("/")[-2:]
    if key_name is None:
        key = os.path.basename(file_path)
    key = f"checkpoints/{group}/{name}/{key}"
    bucket_name = cfg.checkpoint.write_s3_bucket
    s3_client = init_s3(cfg.checkpoint.write_s3_config)
    # Upload the file
    with open(file_path, 'rb') as f:
        s3_client.upload_fileobj(f, bucket_name, key)
    full_s3_path = f"s3://{bucket_name}/{key}"
    logger.info(f'upload_to_s3: {file_path} {full_s3_path} | use time: {time.time()-tic}')

    return full_s3_path


def load_from_s3(file_path, cfg, load_fn):
    """
        ckpt_path example:
            s3://xzeng/checkpoints/2023_0413/vae_kl_5e-1/ckpt/snapshot_epo000163_iter164000.pt
    """
    s3_client = init_s3(cfg.checkpoint.write_s3_config)  # use to test the s3_client can be init
    bucket_name = file_path.split("s3://")[-1].split('/')[0]
    key = file_path.split(f'{bucket_name}/')[-1]
    # logger.info(f"-> try to load s3://{bucket_name}/{key} ")
    tic = time.time()
    for attemp in range(10):
        try:
            # Download the state dict from S3 into memory (as a binary stream)
            with io.BytesIO() as buffer:
                s3_client.download_fileobj(bucket_name, key, buffer)
                buffer.seek(0)

                # Load the state dict into a PyTorch model
                # out = torch.load(buffer, map_location=torch.device("cpu"))
                out = load_fn(buffer)
            break
        except:
            logger.info(f"fail to load s3://{bucket_name}/{key} attemp: {attemp}")
    from torch_utils.dist_utils import is_rank0
    if is_rank0():
        logger.info(f'loaded {file_path} | use time: {time.time()-tic:.1f} sec')
    return out

def load_torch_dict_from_s3(ckpt_path, cfg):
    """
        ckpt_path example:
            s3://xzeng/checkpoints/2023_0413/vae_kl_5e-1/ckpt/snapshot_epo000163_iter164000.pt
    """
    s3_client = init_s3(cfg.checkpoint.write_s3_config)  # use to test the s3_client can be init
    bucket_name = ckpt_path.split("s3://")[-1].split('/')[0]
    key = ckpt_path.split(f'{bucket_name}/')[-1]
    for attemp in range(10):
        try:
            # Download the state dict from S3 into memory (as a binary stream)
            with io.BytesIO() as buffer:
                s3_client.download_fileobj(bucket_name, key, buffer)
                buffer.seek(0)

                # Load the state dict into a PyTorch model
                out = torch.load(buffer, map_location=torch.device("cpu"))
            break
        except:
            logger.info(f"fail to load s3://{bucket_name}/{key} attemp: {attemp}")
    return out

def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

def printarr(*arrs, float_width=6, **kwargs):
    """
    Print a pretty table giving name, shape, dtype, type, and content information for input tensors or scalars.

    Call like: printarr(my_arr, some_other_arr, maybe_a_scalar). Accepts a variable number of arguments.

    Inputs can be:
        - Numpy tensor arrays
        - Pytorch tensor arrays
        - Jax tensor arrays
        - Python ints / floats
        - None

    It may also work with other array-like types, but they have not been tested.

    Use the `float_width` option specify the precision to which floating point types are printed.

    Author: Nicholas Sharp (nmwsharp.com)
    Canonical source: https://gist.github.com/nmwsharp/54d04af87872a4988809f128e1a1d233
    License: This snippet may be used under an MIT license, and it is also released into the public domain. 
             Please retain this docstring as a reference.
    """
    
    frame = inspect.currentframe().f_back
    default_name = "[temporary]"

    ## helpers to gather data about each array
    def name_from_outer_scope(a):
        if a is None:
            return '[None]'
        name = default_name
        for k, v in frame.f_locals.items():
            if v is a:
                name = k
                break
        return name

    def type_strip(type_str):
        return type_str.lstrip('<class ').rstrip('>').replace('torch.', '').strip("'")

    def dtype_str(a):
        if a is None:
            return 'None'
        if isinstance(a, int):
            return 'int'
        if isinstance(a, float):
            return 'float'
        if isinstance(a, list) and len(a)>0:
            return type_strip(str(type(a[0])))
        if hasattr(a, 'dtype'):
            return type_strip(str(a.dtype))
        else:
            return ''
    def shape_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, int):
            return 'scalar'
        if isinstance(a, float):
            return 'scalar'
        if isinstance(a, list):
            return f"[{shape_str(a[0]) if len(a)>0 else '?'}]*{len(a)}"
        if hasattr(a, 'shape'):
            return str(tuple(a.shape))
        else:
            return ''
    def type_str(a):
        return type_strip(str(type(a))) # TODO this is is weird... what's the better way?
    def device_str(a):
        if hasattr(a, 'device'):
            device_str = str(a.device)
            if len(device_str) < 10:
                # heuristic: jax returns some goofy long string we don't want, ignore it
                return device_str
        return ""
    def format_float(x):
        return f"{x:{float_width}g}"
    def minmaxmean_str(a):
        if a is None:
            return ('N/A', 'N/A', 'N/A', 'N/A')
        if isinstance(a, int) or isinstance(a, float): 
            return (format_float(a),)*4

        # compute min/max/mean. if anything goes wrong, just print 'N/A'
        min_str = "N/A"
        try: min_str = format_float(a.min())
        except: pass
        max_str = "N/A"
        try: max_str = format_float(a.max())
        except: pass
        mean_str = "N/A"
        try: mean_str = format_float(a.mean())
        except: pass
        try: median_str = format_float(a.median())
        except:
            try: median_str = format_float(np.median(np.array(a)))
            except: median_str = 'N/A'
        return (min_str, max_str, mean_str, median_str)

    def get_prop_dict(a,k=None):
        minmaxmean = minmaxmean_str(a)
        props = {
            'name' : name_from_outer_scope(a) if k is None else k,
            # 'type' : str(type(a)).replace('torch.',''),
            'dtype' : dtype_str(a),
            'shape' : shape_str(a),
            'type' : type_str(a),
            'device' : device_str(a),
            'min' : minmaxmean[0],
            'max' : minmaxmean[1],
            'mean' : minmaxmean[2],
            'median': minmaxmean[3]
        }
        return props

    try:

        props = ['name', 'type', 'dtype', 'shape', 'device', 'min', 'max', 'mean', 'median']

        # precompute all of the properties for each input
        str_props = []
        for a in arrs:
            str_props.append(get_prop_dict(a))
        for k,a in kwargs.items():
            str_props.append(get_prop_dict(a, k=k))

        # for each property, compute its length
        maxlen = {}
        for p in props: maxlen[p] = 0
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        # if any property got all empty strings, don't bother printing it, remove if from the list
        props = [p for p in props if maxlen[p] > 0]

        # print a header
        header_str = ""
        for p in props:
            prefix =  "" if p == 'name' else " | "
            fmt_key = ">" if p == 'name' else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-"*len(header_str))
            
        # now print the acual arrays
        for strp in str_props:
            for p in props:
                prefix =  "" if p == 'name' else " | "
                fmt_key = ">" if p == 'name' else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end='')
            print("")

    finally:
        del frame

def debug_print_all_tensor_sizes(min_tot_size = 0):
    import gc
    print("---------------------------------------"*3)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if np.prod(obj.size())>=min_tot_size:
                    print(type(obj), obj.size())
        except:
            pass
def print_cpu_usage():
    
    # Get current CPU usage as a percentage
    cpu_usage = psutil.cpu_percent()
    
    # Get current memory usage
    memory_usage = psutil.virtual_memory().used
    
    # Convert memory usage to a human-readable format
    memory_usage_str = psutil._common.bytes2human(memory_usage)
    
    # Print CPU and memory usage
    msg = f"Current CPU usage: {cpu_usage}% | "
    msg += f"Current memory usage: {memory_usage_str}"
    return msg

def calmsize(num_bytes):
    if math.isnan(num_bytes):
        return ''
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num_bytes) < 1024.0:
            return "{:.1f}{}B".format(num_bytes, unit)
        num_bytes /= 1024.0
    return "{:.1f}{}B".format(num_bytes, 'Y')

def readable_size(num_bytes: int) -> str: 
    return calmsize(num_bytes) ## '' if math.isnan(num_bytes) else '{:.1f}'.format(calmsize(num_bytes))

def get_gpu_memory():
    """
    Get the current GPU memory usage for each device as a dictionary
    """
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"])
    output = output.decode("utf-8")
    gpu_memory_values = output.split("\n")[1:-1]
    gpu_memory_values = [int(x.strip().split()[0]) for x in gpu_memory_values]
    gpu_memory = dict(zip(range(len(gpu_memory_values)), gpu_memory_values))
    return gpu_memory

def get_gpu_util():
    """
    Get the current GPU memory usage for each device as a dictionary
    """
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv"])
    output = output.decode("utf-8")
    gpu_memory_values = output.split("\n")[1:-1]
    gpu_memory_values = [int(x.strip().split()[0]) for x in gpu_memory_values]
    gpu_util = dict(zip(range(len(gpu_memory_values)), gpu_memory_values))
    return gpu_util


def print_gpu_usage():
    useage = get_gpu_memory()
    msg = f" | GPU usage: "
    for k, v in useage.items():
        msg += f"{k}: {v} MB "
    # utilization = get_gpu_util()
    # msg + ' | util '
    # for k, v in utilization.items():
    #     msg += f"{k}: {v} % "
    return msg

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def generate_random_string(length):
    # This script will generate a string of 10 random ASCII letters (both lowercase and uppercase).
    # You can adjust the length parameter to fit your needs.
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def check_exist_in_s3(file_path, s3_config):
    s3 = init_s3(s3_config)
    bucket_name, object_name = s3path_to_bucket_key(file_path)

    try:
        s3.head_object(Bucket=bucket_name, Key=object_name)
        return 1
    except:
        logger.info(f'file not found: s3://{bucket_name}/{object_name}')
        return 0

def s3path_to_bucket_key(file_path):
    bucket_name = file_path.split('/')[2]
    object_name = file_path.split(bucket_name + '/')[-1]
    return bucket_name, object_name

def copy_file_to_s3(cfg, file_path_local, file_path_s3):
    # work similar as upload_file_to_s3, but not trying to parse the file path
    # file_path_s3: s3://{bucket}/{key}
    bucket_name, key = s3path_to_bucket_key(file_path_s3)
    tic = time.time()
    s3_client = init_s3(cfg.checkpoint.write_s3_config)

    # Upload the file
    with open(file_path_local, 'rb') as f:
        s3_client.upload_fileobj(f, bucket_name, key)
    full_s3_path = f"s3://{bucket_name}/{key}"
    logger.info(f'copy file: {file_path_local} {full_s3_path} | use time: {time.time()-tic}')
    return full_s3_path