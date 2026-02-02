import normflows as nf
import numpy as np
import torch
from torch import nn
import healpy as hp
import sphericart.torch
from typing import Protocol, Optional, Literal, List, Union, Tuple, Dict

CompileMode = Optional[Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]]

def build_cmlp_diaggaussian_base(input_dim: int, output_dim: int,
                                 hidden_dim: int, num_hidden_layers: int) -> nf.distributions.BaseDistribution:
    context_encoder = BaseMLP(input_dim, output_dim, hidden_dim, num_hidden_layers)
    return ConditionalDiagGaussian(shape=output_dim//2, context_encoder=context_encoder)

def build_c_arqs_flow(base: nf.distributions.BaseDistribution, num_layers: int,
                      latent_dim: int, context_dim: int, num_bins: int,
                      num_hidden_units: int, num_residual_blocks: int) -> nf.ConditionalNormalizingFlow:
        flows = []
        for _ in range(num_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(num_input_channels = latent_dim,
                                                                     num_blocks = num_residual_blocks,
                                                                     num_hidden_channels = num_hidden_units,
                                                                     num_bins = num_bins,
                                                                     num_context_channels = context_dim)]
            flows += [nf.flows.LULinearPermute(latent_dim)]
        return nf.ConditionalNormalizingFlow(base, flows)

class NNDensityInferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model

    def forward(self, 
                source: Optional[torch.Tensor] = None, 
                context: Optional[torch.Tensor] = None,
                n_samples: Optional[int] = None, 
                mode: str = "inference") -> torch.Tensor:
        if mode == "inference":
            if context is None:
                return torch.exp(self._model.log_prob(source))
            else:
                return torch.exp(self._model.log_prob(source, context))
        elif mode == "sampling":
            if context is None:
                return self._model.sample(num_samples=n_samples)[0]
            else:
                return self._model.sample(num_samples=n_samples, context=context)[0]

class ConditionalDiagGaussian(nf.distributions.BaseDistribution):
    def __init__(self, shape: Union[int, List[int], Tuple[int, ...]], context_encoder: nn.Module):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.context_encoder = context_encoder

    def forward(self, num_samples: int=1, context: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=mean.dtype, device=mean.device
        )
        z = mean + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z: torch.Tensor, context: Optional[torch.Tensor]=None) -> torch.Tensor:
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - mean) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p

class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AreaModelProtocol(Protocol):
    @property
    def context_dim(self) -> int: ...
    
    @property
    def compile_mode(self) -> CompileMode: ...
    
    @compile_mode.setter
    def compile_mode(self, value: CompileMode): ...
    
    @property
    def batch_size(self) -> int: ...
    
    @batch_size.setter
    def batch_size(self, value: int): ...

    def evaluate_effective_area(self, *args: torch.Tensor) -> torch.Tensor: ...

class UnpolarizedAreaSphericalHarmonicsExpansion(AreaModelProtocol):
    def __init__(self, area_input: Dict, worker_device: Union[str, int, torch.device],
                 batch_size: int, compile_mode: CompileMode = "max-autotune-no-cudagraphs"):
        self._worker_device = torch.device(worker_device)
        
        self._lmax = area_input['lmax']
        self._poly_degree = area_input['poly_degree']
        self._poly_coeffs = area_input['poly_coeffs']
        
        self._conv_coeffs = self._convert_coefficients().to(self._worker_device)
        self._sh_calculator = sphericart.torch.SphericalHarmonics(self._lmax)
        
        self._compile_mode = compile_mode
        self._compiled_cache = {}
        
        self._update_horner_op()
        
        self._is_cuda = (self._worker_device.type == 'cuda')
        self.batch_size = batch_size
        
        if self._is_cuda:
            self._compute_stream = torch.cuda.Stream(device=self._worker_device)
            self._transfer_stream = torch.cuda.Stream(device=self._worker_device)
            self._transfer_ready = [torch.cuda.Event(), torch.cuda.Event()]
            self._compute_ready = [torch.cuda.Event(), torch.cuda.Event()]
        else:
            self._compute_stream = None
            self._transfer_stream = None
            self._transfer_ready = None
            self._compute_ready = None
    
    def _write_gpu_tensors(self):
        self._gpu_inputs = [
                (torch.empty(self._batch_size, device=self._worker_device),
                 torch.empty(self._batch_size, device=self._worker_device),
                 torch.empty(self._batch_size, device=self._worker_device))
                for _ in range(2)
            ]
        self._gpu_results = [torch.empty(self._batch_size, device=self._worker_device) for _ in range(2)]
    
    @property
    def context_dim(self) -> int:
        return 3
    
    @property
    def compile_mode(self) -> CompileMode:
        return self._compile_mode

    @compile_mode.setter
    def compile_mode(self, value: CompileMode):
        if value != self._compile_mode:
            self._compile_mode = value
            self._update_horner_op()

    def _update_horner_op(self):
        if self._compile_mode is None:
            self._horner_op = self._horner_eval
        else:
            if self._compile_mode not in self._compiled_cache:
                self._compiled_cache[self._compile_mode] = torch.compile(
                    self._horner_eval, 
                    mode=self._compile_mode
                )
            self._horner_op = self._compiled_cache[self._compile_mode]
    
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Batch size must be a positive integer, got {value}")
        self._batch_size = value
        if self._is_cuda:
            self._write_gpu_tensors()
    
    def _convert_coefficients(self) -> torch.Tensor:
        num_sh = (self._lmax + 1)**2
        conv_coeffs = torch.zeros((num_sh, self._poly_degree + 1), dtype=torch.float64)

        for cnt, (l, m) in enumerate((l, m) for l in range(self._lmax + 1) for m in range(-l, l + 1)):
            idx = hp.Alm.getidx(self._lmax, l, abs(m))
            if m == 0:
                conv_coeffs[cnt] = self._poly_coeffs[0, :, idx]
            else:
                fac = np.sqrt(2) * (-1)**m
                val = self._poly_coeffs[0, :, idx] if m > 0 else -self._poly_coeffs[1, :, idx]
                conv_coeffs[cnt] = fac * val
        return conv_coeffs.T
    
    def _horner_eval(self, x: torch.Tensor) -> torch.Tensor:
        x_64 = x.to(torch.float64).unsqueeze(1)
        result = self._conv_coeffs[0].expand(x.shape[0], -1).clone()
        for i in range(1, self._conv_coeffs.size(0)):
            result.mul_(x_64).add_(self._conv_coeffs[i])
        return result.to(torch.float32)
    
    def _compute_spherical_harmonics(self, dir_az: torch.Tensor, dir_polar: torch.Tensor) -> torch.Tensor:
        sin_p = torch.sin(dir_polar)
        xyz = torch.stack((
            sin_p * torch.cos(dir_az),
            sin_p * torch.sin(dir_az),
            torch.cos(dir_polar)
        ), dim=-1)
        return self._sh_calculator(xyz)
    
    @torch.inference_mode()
    def evaluate_effective_area(self, dir_az: torch.Tensor, dir_polar: torch.Tensor, energy_keV: torch.Tensor) -> torch.Tensor:
        N = energy_keV.shape[0]
        
        ei_norm = (torch.log10(energy_keV) / 2 - 1).to(torch.float32)
        result = torch.empty(N, dtype=torch.float32, device="cpu")
        
        def get_batch(start_idx):
            end_idx = min(start_idx + self._batch_size, N)
            return (
                ei_norm[start_idx:end_idx].to(self._worker_device),
                dir_az[start_idx:end_idx].to(self._worker_device),
                dir_polar[start_idx:end_idx].to(self._worker_device)
            )
        
        if self._is_cuda:
            ei_norm = ei_norm.pin_memory()
            result = result.pin_memory()
            
            def enqueue_transfer(slot_idx, start_idx):
                end_idx = min(start_idx + self._batch_size, N)
                size = end_idx - start_idx
                self._gpu_inputs[slot_idx][0][:size].copy_(ei_norm[start_idx:end_idx], non_blocking=True)
                self._gpu_inputs[slot_idx][1][:size].copy_(dir_az[start_idx:end_idx], non_blocking=True)
                self._gpu_inputs[slot_idx][2][:size].copy_(dir_polar[start_idx:end_idx], non_blocking=True)
         
        if self._is_cuda and (N > 0):
            with torch.cuda.stream(self._transfer_stream):
                enqueue_transfer(0, 0)
                self._transfer_ready[0].record(self._transfer_stream)
            
        for i, start in enumerate(range(0, N, self._batch_size)):
            curr_idx = i % 2
            next_idx = (i + 1) % 2
            end = min(start + self._batch_size, N)
            batch_len = end - start
            next_start = start + self._batch_size
            
            if self._is_cuda:
                with torch.cuda.stream(self._compute_stream):
                    self._compute_stream.wait_event(self._transfer_ready[curr_idx])

                    ei_b, az_b, pol_b = [t[:batch_len] for t in self._gpu_inputs[curr_idx]]
                    
                    poly_b = self._horner_op(ei_b)
                    ylm_b = self._compute_spherical_harmonics(az_b, pol_b)
                    
                    torch.sum(poly_b * ylm_b, dim=1, out=self._gpu_results[curr_idx][:batch_len])
                    
                    self._compute_ready[curr_idx].record(self._compute_stream)
                    
                if next_start < N:
                    with torch.cuda.stream(self._transfer_stream):
                        enqueue_transfer(next_idx, next_start)
                        
                        self._transfer_ready[next_idx].record(self._transfer_stream)
                
                with torch.cuda.stream(self._transfer_stream):
                    self._transfer_stream.wait_event(self._compute_ready[curr_idx])
                    result[start:end].copy_(self._gpu_results[curr_idx][:batch_len], non_blocking=True)
            else:
                ei_b, az_b, pol_b = get_batch(start)

                poly_b = self._horner_op(ei_b)
                ylm_b = self._compute_spherical_harmonics(az_b, pol_b)
                result[start:end] = torch.sum(poly_b * ylm_b, dim=1)
            
        if self._is_cuda:
            torch.cuda.synchronize(self._worker_device)
        
        return torch.clamp(result, min=0)

class DensityModelProtocol(Protocol):
    @property
    def context_dim(self) -> int: ...
    
    @property
    def source_dim(self) -> int: ...
    
    @property
    def compile_mode(self) -> CompileMode: ...
    
    @compile_mode.setter
    def compile_mode(self, value: CompileMode): ...
    
    @property
    def batch_size(self) -> int: ...
    
    @batch_size.setter
    def batch_size(self, value: int): ...

    def sample_density(self, *args: torch.Tensor) -> torch.Tensor: ...

    def evaluate_density(self, *args: torch.Tensor) -> torch.Tensor: ...
    
class UnpolarizedDensityCMLPDGaussianCARQSFlow(DensityModelProtocol):
    def __init__(self, density_input: Dict, worker_device: Union[str, int, torch.device],
                 batch_size: int, compile_mode: CompileMode = "default"):
        self._worker_device = torch.device(worker_device)
        
        self._snapshot = density_input["model_state_dict"]
        self._bins = density_input["bins"]
        self._hidden_units = density_input["hidden_units"]
        self._residual_blocks = density_input["residual_blocks"]
        self._total_layers = density_input["total_layers"]
        self._context_size = density_input["context_size"]
        self._latent_size = density_input["latent_size"]
        self._mlp_hidden_units = density_input["mlp_hidden_units"]
        self._mlp_hidden_layers = density_input["mlp_hidden_layers"]
        
        self._compile_mode = compile_mode
        self._compiled_cache = {}
        
        self._eager_model = self._init_base_model()
        self._update_model_op()
        
        self._is_cuda = (self._worker_device.type == 'cuda')
        self.batch_size = batch_size
        
        if self._is_cuda:
            self._compute_stream = torch.cuda.Stream(device=self._worker_device)
            self._transfer_stream = torch.cuda.Stream(device=self._worker_device)
            self._transfer_ready = [torch.cuda.Event(), torch.cuda.Event()]
            self._compute_ready = [torch.cuda.Event(), torch.cuda.Event()]
        else:
            self._compute_stream = None
            self._transfer_stream = None
            self._transfer_ready = None
            self._compute_ready = None

    def _write_gpu_tensors(self):
        self._eval_inputs = [
            tuple(torch.empty(self._batch_size, device=self._worker_device) for _ in range(self.source_dim + self.context_dim))
            for _ in range(2)
        ]
        self._eval_results = [torch.empty(self._batch_size, device=self._worker_device) for _ in range(2)]

        self._sample_inputs = [
            tuple(torch.empty(self._batch_size, device=self._worker_device) for _ in range(self.context_dim))
            for _ in range(2)
        ]

        self._sample_results = [
            (torch.empty((self._batch_size, self._latent_size), device=self._worker_device),
             torch.empty(self._batch_size, dtype=torch.bool, device=self._worker_device))
            for _ in range(2)
        ]
    
    @property
    def context_dim(self) -> int: 
        return 3
    
    @property
    def source_dim(self) -> int: 
        return 4
    
    @property
    def compile_mode(self) -> CompileMode:
        return self._compile_mode
    
    @compile_mode.setter
    def compile_mode(self, value: CompileMode):
        if value != self._compile_mode:
            self._compile_mode = value
            self._update_model_op()
    
    @property
    def batch_size(self) -> int: 
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value: int): 
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Batch size must be a positive integer, got {value}")
        self._batch_size = value
        if self._is_cuda:
            self._write_gpu_tensors()

    def _build_model(self) -> nf.ConditionalNormalizingFlow:
        base = build_cmlp_diaggaussian_base(
            self._context_size, 2 * self._latent_size, self._mlp_hidden_units, self._mlp_hidden_layers
            )
        return build_c_arqs_flow(
            base, self._total_layers, self._latent_size, self._context_size, self._bins, self._hidden_units, self._residual_blocks
            )

    def _init_base_model(self) -> NNDensityInferenceWrapper:
        model = self._build_model()
        
        model.load_state_dict(self._snapshot)
        model = NNDensityInferenceWrapper(model)
        model.eval()
        model.to(self._worker_device)
        
        return model

    def _update_model_op(self):
        if self._compile_mode is None:
            self._model_op = self._eager_model
        else:
            if self._compile_mode not in self._compiled_cache:
                self._compiled_cache[self._compile_mode] = torch.compile(
                    self._eager_model, 
                    mode=self._compile_mode
                )
            self._model_op = self._compiled_cache[self._compile_mode]

    @staticmethod
    def _get_vector(phi_sc: torch.Tensor, theta_sc: torch.Tensor) -> torch.Tensor:
        x = theta_sc[:, 0] * phi_sc[:, 1]
        y = theta_sc[:, 0] * phi_sc[:, 0]
        z = theta_sc[:, 1]
        return torch.stack((x, y, z), dim=-1)

    def _convert_conventions(self, dir_az_sc: torch.Tensor, dir_polar_sc: torch.Tensor, 
                             ei: torch.Tensor, em: torch.Tensor, phi: torch.Tensor, 
                             scatt_az_sc: torch.Tensor, scatt_polar_sc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = em / ei - 1

        source = self._get_vector(dir_az_sc, dir_polar_sc)
        scatter = self._get_vector(scatt_az_sc, scatt_polar_sc)

        dot_product = torch.sum(source * scatter, dim=1)
        phi_geo = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        theta = phi_geo - phi

        xaxis = torch.tensor([1., 0., 0.], device=source.device, dtype=source.dtype)
        pz = -source
        
        px = torch.linalg.cross(pz, xaxis.expand_as(pz))
        px = px / torch.linalg.norm(px, dim=1, keepdim=True)
        
        py = torch.linalg.cross(pz, px)
        py = py / torch.linalg.norm(py, dim=1, keepdim=True)

        proj_x = torch.sum(scatter * px, dim=1)
        proj_y = torch.sum(scatter * py, dim=1)

        zeta = torch.atan2(proj_y, proj_x)
        zeta = torch.where(zeta < 0, zeta + 2 * np.pi, zeta)

        return eps, theta, zeta

    def _inverse_transform_coordinates(self, samples: torch.Tensor, ei: torch.Tensor, 
                                       dir_az_sc: torch.Tensor, dir_pol_sc: torch.Tensor) -> torch.Tensor:
        eps   = -samples[:, 0]
        phi   = samples[:, 1] * np.pi
        theta = (samples[:, 2] - 0.5) * (2 * np.pi)
        zeta  = samples[:, 3] * (2 * np.pi)

        em = ei * (eps + 1)

        phi_geo = theta + phi
        scatter_phf = self._get_vector(torch.stack((torch.sin(zeta), torch.cos(zeta)), dim=1),
                                       torch.stack((torch.sin(np.pi - phi_geo), torch.cos(np.pi - phi_geo)), dim=1))
        source_vec = self._get_vector(dir_az_sc, dir_pol_sc)
        xaxis = torch.tensor([1., 0., 0.], device=self._worker_device, dtype=source_vec.dtype)

        pz = -source_vec
        px = torch.linalg.cross(pz, xaxis.expand_as(pz))
        px = px / torch.linalg.norm(px, dim=1, keepdim=True)
        py = torch.linalg.cross(pz, px)
        py = py / torch.linalg.norm(py, dim=1, keepdim=True)

        basis = torch.stack((px, py, pz), dim=2)
        scatter_scf = torch.bmm(basis, scatter_phf.unsqueeze(-1)).squeeze(-1)

        psi_cds = torch.atan2(scatter_scf[:, 1], scatter_scf[:, 0])
        psi_cds = torch.where(psi_cds < 0, psi_cds + 2 * np.pi, psi_cds)
        chi_cds = torch.acos(torch.clamp(scatter_scf[:, 2], -1.0, 1.0))

        return torch.stack([em, phi, psi_cds, chi_cds], dim=1)

    def _transform_coordinates(self, dir_az: torch.Tensor, dir_pol: torch.Tensor, 
                               ei: torch.Tensor, em: torch.Tensor, phi: torch.Tensor, 
                               scatt_az: torch.Tensor, scatt_pol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        dir_az_sc = torch.stack((torch.sin(dir_az), torch.cos(dir_az)), dim=1)
        dir_pol_sc = torch.stack((torch.sin(dir_pol), torch.cos(dir_pol)), dim=1)
        scatt_az_sc = torch.stack((torch.sin(scatt_az), torch.cos(scatt_az)), dim=1)
        scatt_pol_sc = torch.stack((torch.sin(scatt_pol), torch.cos(scatt_pol)), dim=1)

        eps_raw, theta_raw, zeta_raw = self._convert_conventions(
            dir_az_sc, dir_pol_sc, ei, em, phi, scatt_az_sc, scatt_pol_sc
        )

        jac = 1.0 / (ei * torch.sin(theta_raw + phi) * 4 * np.pi**3)

        ctx = torch.cat([
            (dir_az_sc + 1) / 2, 
            (dir_pol_sc[:, 1:] + 1) / 2, 
            (torch.log10(ei) / 2 - 1).unsqueeze(1)
        ], dim=1)

        src = torch.cat([
            (-eps_raw).unsqueeze(1),
            (phi / np.pi).unsqueeze(1),
            (theta_raw / (2 * np.pi) + 0.5).unsqueeze(1),
            (zeta_raw / (2 * np.pi)).unsqueeze(1)
        ], dim=1)

        return ctx.to(torch.float32), src.to(torch.float32), jac.to(torch.float32)
    
    @staticmethod
    def _valid_samples(samples: torch.Tensor) -> torch.Tensor:
        phi_geo_norm = samples[:, 1] + 2 * samples[:, 2] - 1.0
        valid_mask = (samples[:, 0] <  1.0) & \
                     (samples[:, 1] >  0.0) & (samples[:, 1] <= 1.0) & \
                     (samples[:, 2] >= 0.0) & (samples[:, 2] <= 1.0) & \
                     (samples[:, 3] >= 0.0) & (samples[:, 3] <= 1.0) & \
                     (phi_geo_norm  >  0.0) & (phi_geo_norm  <  1.0)
                     
        return valid_mask

    @torch.inference_mode()
    def sample_density(self, dir_az: torch.Tensor, dir_polar: torch.Tensor, energy_keV: torch.Tensor) -> torch.Tensor:
        N = dir_az.shape[0]
        
        result = torch.empty((N, self._latent_size), dtype=torch.float32, device="cpu")
        failed_mask = torch.zeros(N, dtype=torch.bool, device="cpu")
        
        if self._is_cuda:
            result, failed_mask = result.pin_memory(), failed_mask.pin_memory()
            
            def enqueue_sample_transfer(slot_idx, start_idx):
                end_idx = min(start_idx + self._batch_size, N)
                size = end_idx - start_idx
                self._sample_inputs[slot_idx][0][:size].copy_(energy_keV[start_idx:end_idx], non_blocking=True)
                self._sample_inputs[slot_idx][1][:size].copy_(dir_az[start_idx:end_idx], non_blocking=True)
                self._sample_inputs[slot_idx][2][:size].copy_(dir_polar[start_idx:end_idx], non_blocking=True)
        
        if self._is_cuda and N > 0:
            with torch.cuda.stream(self._transfer_stream):
                enqueue_sample_transfer(0, 0)
                self._transfer_ready[0].record(self._transfer_stream)
        
        for i, start in enumerate(range(0, N, self._batch_size)):
            curr_idx = i % 2
            next_idx = (i + 1) % 2
            end = min(start + self._batch_size, N)
            batch_len = end - start
            next_start = start + self._batch_size
            
            if self._is_cuda:
                with torch.cuda.stream(self._compute_stream):
                    self._compute_stream.wait_event(self._transfer_ready[curr_idx])
                    
                    b_ei, b_az, b_pol = [t[:batch_len] for t in self._sample_inputs[curr_idx]]
                    
                    b_az_sc = torch.stack((torch.sin(b_az), torch.cos(b_az)), dim=1)
                    b_pol_sc = torch.stack((torch.sin(b_pol), torch.cos(b_pol)), dim=1)
                    
                    b_ctx = torch.cat([
                        (b_az_sc + 1) / 2, 
                        (b_pol_sc[:, 1:] + 1) / 2, 
                        (torch.log10(b_ei) / 2 - 1).unsqueeze(1)
                    ], dim=1).to(torch.float32)
                    
                    b_latent = self._model_op(context=b_ctx, mode="sampling", n_samples=batch_len)
                    
                    self._sample_results[curr_idx][0][:batch_len] = self._inverse_transform_coordinates(
                        b_latent, b_ei, b_az_sc, b_pol_sc
                    )
                    self._sample_results[curr_idx][1][:batch_len] = ~self._valid_samples(b_latent)
                    
                    self._compute_ready[curr_idx].record(self._compute_stream)

                if next_start < N:
                    with torch.cuda.stream(self._transfer_stream):
                        enqueue_sample_transfer(next_idx, next_start)
                        self._transfer_ready[next_idx].record(self._transfer_stream)

                with torch.cuda.stream(self._transfer_stream):
                    self._transfer_stream.wait_event(self._compute_ready[curr_idx])
                    
                    result[start:end].copy_(self._sample_results[curr_idx][0][:batch_len], non_blocking=True)
                    failed_mask[start:end].copy_(self._sample_results[curr_idx][1][:batch_len], non_blocking=True)
            else:
                b_ei = energy_keV[start:end].to(self._worker_device)
                b_az, b_pol = dir_az[start:end].to(self._worker_device), dir_polar[start:end].to(self._worker_device)
                
                b_az_sc = torch.stack((torch.sin(b_az), torch.cos(b_az)), dim=1)
                b_pol_sc = torch.stack((torch.sin(b_pol), torch.cos(b_pol)), dim=1)
                b_ctx = torch.cat([
                    (b_az_sc + 1) / 2, (b_pol_sc[:, 1:] + 1) / 2, 
                    (torch.log10(b_ei) / 2 - 1).unsqueeze(1)
                ], dim=1).to(torch.float32)
                
                b_samples = self._model_op(context=b_ctx, mode="sampling", n_samples=batch_len)
                result[start:end] = self._inverse_transform_coordinates(b_samples, b_ei, b_az_sc, b_pol_sc)
                failed_mask[start:end] = ~self._valid_samples(b_samples)

        if self._is_cuda:
            torch.cuda.synchronize(self._worker_device)

        if torch.any(failed_mask):
            result[failed_mask] = self.sample_density(
                dir_az[failed_mask], dir_polar[failed_mask], energy_keV[failed_mask]
            )

        return result
    
    @torch.inference_mode()
    def evaluate_density(
        self, dir_az: torch.Tensor, dir_polar: torch.Tensor,
        energy_keV: torch.Tensor, menergy_keV: torch.Tensor,
        scatt_angle: torch.Tensor, scatt_az: torch.Tensor,
        scatt_polar: torch.Tensor) -> torch.Tensor:
        
        N = dir_az.shape[0]
        result = torch.empty(N, dtype=torch.float32, device="cpu")
        
        if self._is_cuda:
            result = result.pin_memory()
            
            def enqueue_eval_transfer(slot_idx, start_idx):
                end_idx = min(start_idx + self._batch_size, N)
                size = end_idx - start_idx
                tensors = [dir_az, dir_polar, energy_keV, menergy_keV, scatt_angle, scatt_az, scatt_polar]
                for i in range(self.source_dim + self.context_dim):
                    self._eval_inputs[slot_idx][i][:size].copy_(tensors[i][start_idx:end_idx], non_blocking=True)
        
        if self._is_cuda and N > 0:
            with torch.cuda.stream(self._transfer_stream):
                enqueue_eval_transfer(0, 0)
                self._transfer_ready[0].record(self._transfer_stream)
        
        for i, start in enumerate(range(0, N, self._batch_size)):
            curr_idx = i % 2
            next_idx = (i + 1) % 2
            end = min(start + self._batch_size, N)
            batch_len = end - start
            next_start = start + self._batch_size
            
            if self._is_cuda:
                with torch.cuda.stream(self._compute_stream):
                    self._compute_stream.wait_event(self._transfer_ready[curr_idx])
                    
                    ctx, src, jac = self._transform_coordinates(*[t[:batch_len] for t in self._eval_inputs[curr_idx]])
                    
                    torch.mul(self._model_op(src, ctx, mode="inference"), jac, out=self._eval_results[curr_idx][:batch_len])
                    
                    self._compute_ready[curr_idx].record(self._compute_stream)

                if next_start < N:
                    with torch.cuda.stream(self._transfer_stream):
                        enqueue_eval_transfer(next_idx, next_start)
                        
                        self._transfer_ready[next_idx].record(self._transfer_stream)

                with torch.cuda.stream(self._transfer_stream):
                    self._transfer_stream.wait_event(self._compute_ready[curr_idx])
                    
                    result[start:end].copy_(self._eval_results[curr_idx][:batch_len], non_blocking=True)
            else:
                b_az, b_pol = dir_az[start:end].to(self._worker_device), dir_polar[start:end].to(self._worker_device)
                b_ei, b_em = energy_keV[start:end].to(self._worker_device), menergy_keV[start:end].to(self._worker_device)
                b_phi = scatt_angle[start:end].to(self._worker_device)
                b_s_az, b_s_pol = scatt_az[start:end].to(self._worker_device), scatt_polar[start:end].to(self._worker_device)
                
                ctx, src, jac = self._transform_coordinates(b_az, b_pol, b_ei, b_em, b_phi, b_s_az, b_s_pol)
                result[start:end] = self._model_op(src, ctx, mode="inference") * jac

        if self._is_cuda:
            torch.cuda.synchronize(self._worker_device)
        return result