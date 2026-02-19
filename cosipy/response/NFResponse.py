from typing import List, Union


from importlib.util import find_spec

if find_spec("torch") is None:
    raise RuntimeError("Install cosipy with [ml] optional package to use this feature.")

import torch
import torch.multiprocessing as mp
from .nf_response_helper import *


def cuda_cleanup_task(_) -> bool:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

def update_worker_settings(args: Tuple[str, Union[int, CompileMode]]):
    attr, value = args
    global area_module
    global density_module
    
    if attr == 'area_batch_size':
        area_module.batch_size = value
    elif attr == 'density_batch_size':
        density_module.batch_size = value
    elif attr == 'area_compile_mode':
        area_module.compile_mode = value
    elif attr == 'density_compile_mode':
        density_module.compile_mode = value

def init_worker(device_queue: mp.Queue, major_version: int, area_input: Dict,
                density_input: Dict, area_batch_size: int, density_batch_size: int,
                area_compile_mode: CompileMode, density_compile_mode: CompileMode):
    global area_module
    global density_module
    global worker_device
    
    worker_device = torch.device(device_queue.get())
    if worker_device.type == 'cuda':
        torch.cuda.set_device(worker_device)
    
    area_module = AreaApproximation(major_version, area_input, worker_device, area_batch_size, area_compile_mode)
    density_module = DensityApproximation(major_version, density_input, worker_device, density_batch_size, density_compile_mode)

def evaluate_area_task(args: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    global area_module
    context, indices = args
    
    sub_context = context[indices, :]
    if torch.device(worker_device).type == 'cuda':
        sub_context = sub_context.pin_memory()
    
    return area_module.evaluate_effective_area(sub_context)

def evaluate_density_task(args: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    global density_module
    context, source, indices = args
    
    sub_context = context[indices, :]
    sub_source = source[indices, :]
    if torch.device(worker_device).type == 'cuda':
        sub_context = sub_context.pin_memory()
        sub_source = sub_source.pin_memory()
    
    return density_module.evaluate_density(sub_context, sub_source)

def sample_density_task(args: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    global density_module
    context, indices = args
    
    sub_context = context[indices, :]
    if torch.device(worker_device).type == 'cuda':
        sub_context = sub_context.pin_memory()
    
    return density_module.sample_density(sub_context)

class DensityApproximation:
    def __init__(self, major_version: int, density_input: Dict, worker_device: Union[str, int, torch.device], batch_size: int, compile_mode: CompileMode):
        self._major_version = major_version
        self._worker_device = worker_device
        self._density_input = density_input
        self._batch_size = batch_size
        self._compile_mode = compile_mode
        
        self._setup_model()

    def _setup_model(self):
        version_map: Dict[int, DensityModelProtocol] = {
            1: UnpolarizedDensityCMLPDGaussianCARQSFlow(self._density_input, self._worker_device, self._batch_size, self._compile_mode),
        }
        if self._major_version not in version_map:
            raise ValueError(f"Unsupported major version {self._major_version} for Density Approximation")
        else:    
            self._model = version_map[self._major_version]
            self._expected_context_dim = self._model.context_dim
            self._expected_source_dim = self._model.source_dim
    
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

class AreaApproximation:
    def __init__(self, major_version: int, area_input: Dict, worker_device: Union[str, int, torch.device], batch_size: int, compile_mode: CompileMode):
        self._major_version = major_version
        self._worker_device = worker_device
        self._area_input = area_input
        self._batch_size = batch_size
        self._compile_mode = compile_mode
        
        self._setup_model()

    def _setup_model(self):
        version_map: Dict[int, AreaModelProtocol] = {
            1: UnpolarizedAreaSphericalHarmonicsExpansion(self._area_input, self._worker_device, self._batch_size, self._compile_mode),
        }
        if self._major_version not in version_map:
            raise ValueError(f"Unsupported major version {self._major_version} for Effective Area Approximation")
        else:    
            self._model = version_map[self._major_version]
            self._expected_context_dim = self._model.context_dim
    
    def evaluate_effective_area(self, context: torch.Tensor) -> torch.Tensor:
        dim_context = context.shape[1]
        
        if dim_context != self._expected_context_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_context_dim} features, but context has {dim_context}."
            )
        
        list_context = [context[:, i] for i in range(dim_context)]
        
        return self._model.evaluate_effective_area(*list_context)

class NFResponse:
    def __init__(self, path_to_model: str, area_batch_size: int = 100_000, density_batch_size: int = 100_000,
                 devices: Optional[List[Union[str, int, torch.device]]] = None,
                 area_compile_mode: CompileMode = "max-autotune-no-cudagraphs", density_compile_mode: CompileMode = "default"):
        ckpt = torch.load(path_to_model, map_location=torch.device('cpu'), weights_only=False)
        
        required_keys = ['version', 'is_polarized', 'density_input', 'area_input']
        
        for key in required_keys:
            if key not in ckpt:
                raise KeyError(
                    f"Invalid Checkpoint: Metadata key '{key}' not found in {path_to_model}. "
                    f"Ensure you saved the model as a dictionary, not just the state_dict."
                )
        
        self._version = ckpt['version']
        self._major_version = int(self._version.split('.')[0])
        self._is_polarized = ckpt['is_polarized']
        self._density_input = ckpt['density_input']
        self._area_input = ckpt['area_input']
        
        self._pool = None
        self._has_cuda = False
        self._ctx = mp.get_context("spawn")
        
        self.area_batch_size = area_batch_size
        self.density_batch_size = density_batch_size
        self._area_compile_mode = area_compile_mode
        self._density_compile_mode = density_compile_mode
        
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
    def is_polarized(self) -> bool:
        return self._is_polarized
    
    @property
    def area_batch_size(self) -> int:
        return self._area_batch_size
    
    @area_batch_size.setter
    def area_batch_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("area_batch_size must be a positive integer")
        self._area_batch_size = value
        self._update_worker_config('area_batch_size', value)
    
    @property
    def density_batch_size(self) -> int:
        return self._density_batch_size
    
    @density_batch_size.setter
    def density_batch_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("density_batch_size must be a positive integer")
        self._density_batch_size = value
        self._update_worker_config('density_batch_size', value)
    
    @property
    def area_compile_mode(self) -> CompileMode: return self._area_compile_mode
    
    @area_compile_mode.setter
    def area_compile_mode(self, value: CompileMode):
        self._area_compile_mode = value
        self._update_worker_config('area_compile_mode', value)
    
    @property
    def density_compile_mode(self) -> CompileMode: return self._density_compile_mode
    
    @density_compile_mode.setter
    def density_compile_mode(self, value: CompileMode):
        self._density_compile_mode = value
        self._update_worker_config('density_compile_mode', value)
    
    def _update_worker_config(self, attr: str, value: Union[int, CompileMode]):
        if self._pool is not None:
            self._pool.map(update_worker_settings, [(attr, value)] * self._num_workers)
    
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
            initializer=init_worker,
            initargs=(device_queue, self._major_version, self._area_input, self._density_input,
                      self._area_batch_size, self._density_batch_size,
                      self._area_compile_mode, self._density_compile_mode),
        )
        
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
            #if self._has_cuda and not context.is_pinned():
            #    context = context.pin_memory()

            n_data = context.shape[0]
            indices = torch.tensor_split(torch.arange(n_data), self._num_workers)
            
            tasks = [(context, idx) for idx in indices]
            results = self._pool.map(sample_density_task, tasks)
            
            return torch.cat(results, dim=0)

        finally:
            if temp_pool:
                self.shutdown_compute_pool()
    
    def evaluate_effective_area(self, context: torch.Tensor, devices: Optional[List[Union[str, int, torch.device]]]=None) -> torch.Tensor:
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
            #if self._has_cuda and not context.is_pinned():
            #    context = context.pin_memory()

            n_data = context.shape[0]
            indices = torch.tensor_split(torch.arange(n_data), self._num_workers)
            
            tasks = [(context, idx) for idx in indices]
            results = self._pool.map(evaluate_area_task, tasks)
            
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
            
            #if self._has_cuda:
            #    if not context.is_pinned():
            #        context = context.pin_memory()
            #    if not source.is_pinned():
            #        source = source.pin_memory()
            
            n_data = context.shape[0]
            indices = torch.tensor_split(torch.arange(n_data), self._num_workers)
            
            tasks = [(context, source, idx) for idx in indices]
            results = self._pool.map(evaluate_density_task, tasks)
            
            return torch.cat(results, dim=0)

        finally:
            if temp_pool:
                self.shutdown_compute_pool()