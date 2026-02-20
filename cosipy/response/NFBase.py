from typing import List, Union, Optional
from pathlib import Path
from abc import ABC, abstractmethod


from importlib.util import find_spec

if find_spec("torch") is None:
    raise RuntimeError("Install cosipy with [ml] optional package to use this feature.")

import torch
import torch.multiprocessing as mp
from .NFModels import *
import cosipy.response.NFWorkerState as NFWorkerState


class DensityApproximation(ABC):
    def __init__(self, major_version: int, density_input: Dict, worker_device: Union[str, int, torch.device], batch_size: int, compile_mode: CompileMode):
        self._major_version = major_version
        self._worker_device = worker_device
        self._density_input = density_input
        self._batch_size = batch_size
        self._compile_mode = compile_mode
        
        self._model: DensityModel
        self._expected_context_dim: int
        self._expected_source_dim: int
        
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        ...
    
    def evaluate_density(self, context: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        dim_context = context.shape[1]
        dim_source = source.shape[1]
        
        if dim_context != self._expected_context_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_context_dim} features, but context has {dim_context}."
            )
        elif dim_source != self._expected_source_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_source_dim} features, but source has {dim_source}."
            )
            
        list_context = [context[:, i] for i in range(dim_context)]
        list_source = [source[:, i] for i in range(dim_source)]
        
        return self._model.evaluate_density(*list_context, *list_source)
    
    def sample_density(self, context: torch.Tensor) -> torch.Tensor:
        dim_context = context.shape[1]
        
        if dim_context != self._expected_context_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_context_dim} features, but context has {dim_context}."
            )
            
        list_context = [context[:, i] for i in range(dim_context)]
        
        return self._model.sample_density(*list_context)

def cuda_cleanup_task(_) -> bool:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

def update_density_worker_settings(args: Tuple[str, Union[int, CompileMode]]):
    attr, value = args
    
    if attr == 'density_batch_size':
        NFWorkerState.density_module._model.batch_size = value
    elif attr == 'density_compile_mode':
        NFWorkerState.density_module._model.compile_mode = value

def init_density_worker(device_queue: mp.Queue, major_version: int,
                        density_input: Dict, density_batch_size: int,
                        density_compile_mode: CompileMode, density_approximation: DensityApproximation):
    
    NFWorkerState.worker_device = torch.device(device_queue.get())
    if NFWorkerState.worker_device.type == 'cuda':
        torch.cuda.set_device(NFWorkerState.worker_device)
    
    NFWorkerState.density_module = density_approximation(major_version, density_input, NFWorkerState.worker_device, density_batch_size, density_compile_mode)

def evaluate_density_task(args: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    context, source, indices = args
    
    sub_context = context[indices, :]
    sub_source = source[indices, :]
    if torch.device(NFWorkerState.worker_device).type == 'cuda':
        sub_context = sub_context.pin_memory()
        sub_source = sub_source.pin_memory()
    
    return NFWorkerState.density_module.evaluate_density(sub_context, sub_source)

def sample_density_task(args: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    context, indices = args
    
    sub_context = context[indices, :]
    if torch.device(NFWorkerState.worker_device).type == 'cuda':
        sub_context = sub_context.pin_memory()
    
    return NFWorkerState.density_module.sample_density(sub_context)

class NFBase():
    def __init__(self, path_to_model: Union[str, Path], update_worker, pool_worker, density_batch_size: int = 100_000,
                 devices: Optional[List[Union[str, int, torch.device]]] = None, density_compile_mode: CompileMode = "default",
                 additional_required_keys: List[str] = None):
        self._ckpt = torch.load(str(path_to_model), map_location=torch.device('cpu'), weights_only=False)
        
        required_keys = ['version', 'density_input'] + (additional_required_keys or [])
        
        for key in required_keys:
            if key not in self._ckpt:
                raise KeyError(
                    f"Invalid Checkpoint: Metadata key '{key}' not found in {str(path_to_model)}. "
                    f"Ensure you saved the model as a dictionary, not just the state_dict."
                )
        
        self._version = self._ckpt['version']
        self._major_version = int(self._version.split('.')[0])
        self._density_input = self._ckpt['density_input']
        
        self._pool = None
        self._has_cuda = False
        self._num_workers = 0
        self._ctx = mp.get_context("spawn")
        self._pool_worker = pool_worker
        self._update_worker = update_worker
        
        self.density_batch_size = density_batch_size
        self.density_compile_mode = density_compile_mode
        
        self._update_pool_arguments()
        
        if devices is not None:
            self.devices = devices
        else:
            self._devices = []
            
    def __del__(self):
        self.shutdown_compute_pool()
    
    @property
    def devices(self) -> List[Union[str, int, torch.device]]:
        return self._devices
    
    @devices.setter
    def devices(self, value: List[Union[str, int, torch.device]]):
        if not isinstance(value, list):
            raise ValueError("devices must be a list of device identifiers")
        self._devices = value
    
    @property
    def density_batch_size(self) -> int:
        return self._density_batch_size
    
    @density_batch_size.setter
    def density_batch_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("density_batch_size must be a positive integer")
        self._density_batch_size = value
        self._update_pool_arguments()
        self._update_worker_config('density_batch_size', value)
    
    @property
    def density_compile_mode(self) -> CompileMode: return self._density_compile_mode
    
    @density_compile_mode.setter
    def density_compile_mode(self, value: CompileMode):
        self._density_compile_mode = value
        self._update_pool_arguments()
        self._update_worker_config('density_compile_mode', value)
    
    def _update_worker_config(self, attr: str, value: Union[int, CompileMode]):
        if self._pool is not None:
            self._pool.map(self._update_worker, [(attr, value)] * self._num_workers)
    
    def _update_pool_arguments(self):
        self._pool_arguments = [
            getattr(self, "_major_version", None),
            getattr(self, "_density_input", None),
            getattr(self, "_density_batch_size", None),
            getattr(self, "_density_compile_mode", None),
            ]
    
    def clean_compute_pool(self):
        if self._pool:
            self._pool.map(cuda_cleanup_task, range(self._num_workers))
        
    def shutdown_compute_pool(self):
        if self._pool:
            self._pool.close()
            self._pool.join()
        
        self._num_workers = 0
        self._pool = None
        self._has_cuda = None
    
    def init_compute_pool(self, devices: Optional[List[Union[str, int, torch.device]]]=None):
        active_devices = devices if devices is not None else self._devices
        
        if not active_devices:
            raise RuntimeError("Cannot initialize pool: no devices provided as argument or set as fallback.")
        
        if self._pool:
            print("Warning: Pool already initialized. Shutting down old pool first.")
            self.shutdown_compute_pool()
        
        self._num_workers = len(active_devices)
        self._has_cuda = any(torch.device(d).type == 'cuda' for d in active_devices)
        
        device_queue = self._ctx.Queue()
        for d in active_devices:
            device_queue.put(d)
        
        self._pool = self._ctx.Pool(
            processes=self._num_workers,
            initializer=self._pool_worker,
            initargs=(device_queue, *self._pool_arguments),
        )
    
    def sample_density(self, context: torch.Tensor, devices: Optional[List[Union[str, int, torch.device]]]=None) -> torch.Tensor:
        temp_pool = False
        if self._pool is None:
            target_devices = devices if devices is not None else self._devices
            if not target_devices:
                raise RuntimeError("No compute pool initialized and no devices provided/set.")
            self.init_compute_pool(target_devices)
            temp_pool = True
        
        try:
            if not context.is_shared():
                context.share_memory_()

            n_data = context.shape[0]
            indices = torch.tensor_split(torch.arange(n_data), self._num_workers)
            
            tasks = [(context, idx) for idx in indices]
            results = self._pool.map(sample_density_task, tasks)
            
            return torch.cat(results, dim=0)

        finally:
            if temp_pool:
                self.shutdown_compute_pool()
    
    def evaluate_density(self, context: torch.Tensor, source: torch.Tensor, devices: Optional[List[Union[str, int, torch.device]]]=None) -> torch.Tensor:
        temp_pool = False
        if self._pool is None:
            target_devices = devices if devices is not None else self._devices
            if not target_devices:
                raise RuntimeError("No compute pool initialized and no devices provided/set.")
            
            self.init_compute_pool(target_devices)
            temp_pool = True
        
        try:    
            if not context.is_shared(): context.share_memory_()
            if not source.is_shared(): source.share_memory_()
            
            n_data = context.shape[0]
            indices = torch.tensor_split(torch.arange(n_data), self._num_workers)
            
            tasks = [(context, source, idx) for idx in indices]
            results = self._pool.map(evaluate_density_task, tasks)
            
            return torch.cat(results, dim=0)

        finally:
            if temp_pool:
                self.shutdown_compute_pool()
