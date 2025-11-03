# ACWNN: Adaptive Constructive Wavelet Neural Network (No Pruning Version)
# This is the baseline ACWNN without pruning/freezing mechanisms
# Key features: Optional disturbance testing, adaptive dwell time, continuous layer expansion
# Users can toggle disturbance testing on/off via configuration

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torchdiffeq import odeint
import openpyxl
import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import math
from typing import Callable, Optional, Dict, Any
import psutil
import gc
from datetime import datetime


class AutomaticTrajectoryProcessor:
    """Process trajectory function and its derivatives automatically"""
    
    def __init__(self, trajectory_func: Callable, system_order: int, device: str):
        self.trajectory_func = trajectory_func
        self.system_order = system_order
        self.device = device
        self.derivatives_cache = {}
        self.use_manual = False
        self.max_cache_size = 10
        
    def set_manual_derivatives(self, derivatives_func: Callable):
        """Set manually computed derivatives function"""
        self.manual_derivatives = derivatives_func
        self.use_manual = True
        
    def compute_derivatives(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute trajectory and all necessary derivatives"""
        if self.use_manual:
            return self.manual_derivatives(t)
        
        t_key = t.item() if t.numel() == 1 else tuple(t.cpu().numpy())
        
        if t_key in self.derivatives_cache:
            return self.derivatives_cache[t_key]
        
        t_var = t.clone().detach().requires_grad_(True)
        xr = self.trajectory_func(t_var)
        
        derivatives = {'xr': xr.detach()}
        current_deriv = xr
        
        for order in range(1, self.system_order + 1):
            grad = torch.autograd.grad(
                current_deriv, t_var, 
                create_graph=True, 
                retain_graph=True
            )[0]
            derivatives[f'xr_dot{order}'] = grad.detach()
            current_deriv = grad
        
        if len(self.derivatives_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.derivatives_cache))
            del self.derivatives_cache[oldest_key]
        
        self.derivatives_cache[t_key] = derivatives
        return derivatives


class AdaptiveWaveletControl(nn.Module):
    """
    ACWNN: Adaptive Constructive Wavelet Neural Network (No Pruning Version)
    
    This is the baseline comparison method without pruning/freezing mechanisms.
    Key differences from full ACWNN:
    - No pruning stage (always remains in Stage 1)
    - No weight freezing
    - No parameter removal
    - Continuous network expansion based on error threshold
    
    Optional disturbance testing feature:
    - Users can enable/disable disturbance via 'disturbance_enabled' parameter
    - Configurable time window and magnitude
    """
    
    def __init__(self, config, device):
        super().__init__()
        # System configuration
        self.config = config
        self.j = config['j_init']
        self.mu = config['mu']
        self.tran_r = config['tran_r']
        self.max_layers = config['max_layers']
        self.device = device
        
        # Automated system configuration
        self.system_order = config.get('system_order', 4)
        self.dimension = self.system_order
        self.lambda_c = config.get('lambda_c', 2.0)
        self.beta = config.get('beta', 10.0)
        
        # Disturbance configuration (optional feature)
        self.disturbance_enabled = config.get('disturbance_enabled', False)
        self.disturbance_start_time = config.get('disturbance_start_time', 240.0)
        self.disturbance_end_time = config.get('disturbance_end_time', 360.0)
        self.disturbance_magnitude = config.get('disturbance_magnitude', 0.5)
        
        # Auto-compute control coefficients
        self._compute_control_coefficients()
        
        # Setup trajectory and nonlinear functions
        self._setup_trajectory_processor(config)
        self._setup_nonlinear_function(config)

        # Output directory
        self.plot_dir = "loss_plots"
        os.makedirs(self.plot_dir, exist_ok=True)

        # Cache control parameters
        self.max_cache_size = config.get('max_cache_size', 10000)
        self.max_cacheable_j = config.get('max_cacheable_j', 5)
                
        # Adaptive dwell time parameters
        self.N_d = config.get('N_d', 3)
        self.N_counter = 0
        self.dwell_time = config['base_interval']
        self.current_dwell_start = 0.0
        self.dwell_losses = []
        
        # Single error threshold (no pruning in this version)
        self.delta_ac = config['error_threshold']
        
        # Stage control (always Stage 1 for no-pruning version)
        self.stage = 1
        
        # Precomputed constants
        self.max_stage = int(1 / self.mu)
        self.learning_rate = 0.0001
        self.scaling = 0.5
        self.inv_scaling = 2.0
        
        # Training data records
        self.training_records = defaultdict(list)
        
        # Parameter management
        self.weights = nn.ParameterList()
        self.init_weights = {}
        self.centers_list = []
        self.active_indices = []
        self.active_stages = []
        self.parent_stage_map = {}
        
        # Cache system
        self.wavelet_cache = {}
        self.cache_x_est = None
        self.power_cache = {}
        
        # ADAM optimizer state
        self.adam_m = []
        self.adam_v = []
        self.adam_t = 0
        
        # Precomputation
        self._precompute_grid_templates()
        self._precompute_powers()
        self._init_base_layer()
        self._init_adam_states()
        
        # Initialization
        self.last_print_time = -0.01
        self.stop_flag = False
        
        # Memory monitoring
        self.process = psutil.Process()
        self.initial_rss = self.process.memory_info().rss / 1024**3
        self.memory_check_points = []
        print(f"[Memory] Initial RSS: {self.initial_rss:.2f}GB")
        
        if self.disturbance_enabled:
            print(f"\n[Disturbance] Enabled from t={self.disturbance_start_time} to t={self.disturbance_end_time}")
            print(f"[Disturbance] Magnitude: {self.disturbance_magnitude}")
    
    def _compute_control_coefficients(self):
        """Automatically compute control coefficients based on system order"""
        d = self.system_order
        λ = self.lambda_c
        
        import math
        
        self.control_coeffs = []
        
        for k in range(d):
            if k <= d-1:
                n = d - 1
                comb_value = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
            else:
                comb_value = 0
                
            coeff = comb_value * (λ ** (d - 1 - k))
            self.control_coeffs.append(coeff)
        
        print(f"System order: {d}, λ: {λ}") 
        print(f"Control coefficients: {self.control_coeffs}")
    
    def _setup_trajectory_processor(self, config):
        """Setup trajectory processor"""
        trajectory_func = config.get('trajectory_function')
        
        if trajectory_func is None:
            trajectory_func = lambda t: 0.5 * torch.sin(t) ** 2
            
            def manual_derivatives(t):
                sin_t = torch.sin(t)
                cos_t = torch.cos(t)
                sin_2t = torch.sin(2*t)
                cos_2t = torch.cos(2*t)
                
                return {
                    'xr': 0.5 * sin_t * sin_t,
                    'xr_dot1': sin_t * cos_t,
                    'xr_dot2': cos_2t,
                    'xr_dot3': -2 * sin_2t,
                    'xr_dot4': -4 * cos_2t
                }
            
            self.trajectory_processor = AutomaticTrajectoryProcessor(
                trajectory_func, self.system_order, self.device
            )
            self.trajectory_processor.set_manual_derivatives(manual_derivatives)
        else:
            self.trajectory_processor = AutomaticTrajectoryProcessor(
                trajectory_func, self.system_order, self.device
            )
    
    def _setup_nonlinear_function(self, config):
        """Setup nonlinear function"""
        nonlinear_func = config.get('nonlinear_function')
        
        if nonlinear_func is None:
            self.compute_nonlinear = self._original_nonlinear
        else:
            self.compute_nonlinear = nonlinear_func
    
    def _original_nonlinear(self, x_est):
        """Original system nonlinear function"""
        if self.system_order >= 4:
            x0, x1, x2, x3 = x_est[0], x_est[1], x_est[2], x_est[3]
            x_sum = x0 + 2*(x1 + x2 + x3)
            x_partial = x0 + 2*x1 + x2
            x_tail = x1 + 2*x2 + x3
            return -x_sum + (1 - x_partial*x_partial) * x_tail
        else:
            return -torch.sum(x_est[:self.system_order])
    
    def _precompute_powers(self):
        """Precompute commonly used powers"""
        for j in range(10):
            self.power_cache[j] = self.inv_scaling ** j

    def _precompute_grid_templates(self):
        """Precompute grid generation templates"""
        self.grid_cache = {}
        self.mesh_cache = {}
        
        max_j_to_cache = min(10, self.max_cacheable_j)
        
        for j in range(max_j_to_cache):
            grid_size = self.tran_r * 2**j + 1
            
            if grid_size <= 100:
                grid = torch.arange(grid_size, device=self.device, dtype=torch.float32)
                self.grid_cache[grid_size] = grid
                
                print(f"Pre-cached grid for j={j}, grid_size={grid_size}")
    
    def _init_adam_states(self):
        """Initialize ADAM optimizer states"""
        for w in self.weights:
            self.adam_m.append(torch.zeros_like(w.data))
            self.adam_v.append(torch.zeros_like(w.data))
    
    def _init_base_layer(self):
        """Initialize base layer"""
        centers = self._generate_centers(self.j)
        n_centers = len(centers)
        
        self.centers_list.append(centers)
        self.active_indices.append(torch.arange(n_centers, device=self.device))
        self.active_stages.append(self.max_stage - 1)
        
        weight_size = 2 * n_centers
        self.weights.append(nn.Parameter(torch.zeros(weight_size, device=self.device)))
        self.init_weights[0] = torch.zeros(weight_size, device=self.device)
    
    @torch.jit.unused
    def _generate_centers(self, j):
        """Generate center points"""
        grid_size = self.tran_r * 2**j + 1
        
        if grid_size in self.mesh_cache:
            return self.mesh_cache[grid_size]
        
        if grid_size in self.grid_cache:
            grid = self.grid_cache[grid_size]
        else:
            grid = torch.arange(grid_size, device=self.device, dtype=torch.float32)
            if grid_size <= 100:
                self.grid_cache[grid_size] = grid
        
        grids = [grid] * self.dimension
        mesh = torch.meshgrid(grids, indexing='ij')
        centers = torch.stack(mesh, dim=-1).reshape(-1, self.dimension).contiguous()
        
        n_centers = len(centers)
        should_cache = (
            j <= self.max_cacheable_j and 
            n_centers <= self.max_cache_size and
            grid_size <= 100
        )
        
        if should_cache:
            self.mesh_cache[grid_size] = centers
            print(f"Cached centers for j={j}, grid_size={grid_size}, n_centers={n_centers}")
        
        return centers
    
    def _compute_wavelet_terms_fast(self, x_est):
        """Fast computation of wavelet function values for all layers"""
        need_recompute = (self.cache_x_est is None or 
                        not torch.allclose(self.cache_x_est, x_est, rtol=1e-9, atol=1e-9))
        
        if need_recompute:
            if hasattr(self, 'wavelet_cache') and self.wavelet_cache:
                for layer_idx in list(self.wavelet_cache.keys()):
                    if layer_idx in self.wavelet_cache:
                        del self.wavelet_cache[layer_idx]
                self.wavelet_cache.clear()
            
            if self.cache_x_est is not None:
                del self.cache_x_est
            
            self.wavelet_cache = {}
            self.cache_x_est = x_est.clone()
            
            for layer_idx in range(len(self.weights)):
                if layer_idx == 0:
                    j = self.j
                    centers = self.centers_list[0]
                    phi, psi = self._wavelet_function_batch(x_est, centers, j, True)
                    combined = torch.cat([phi, psi])
                    
                    self.wavelet_cache[0] = {'phi': phi, 'psi': psi, 'j': j, 'combined': combined}
                else:
                    j = self.j + layer_idx
                    active_idx = self.active_indices[layer_idx]
                    if len(active_idx) > 0:
                        centers = self.centers_list[layer_idx][active_idx]
                        _, psi = self._wavelet_function_batch(x_est, centers, j, False)
                        
                        self.wavelet_cache[layer_idx] = {'psi': psi, 'j': j}
        
        return self.wavelet_cache

    def _wavelet_function_batch(self, x, centers, j, is_base_layer):
        """Batch computation of wavelet functions"""
        scale_factor = self.power_cache.get(j, self.inv_scaling ** j)
        
        n_centers = len(centers)
        max_batch_size = 5000
        
        if n_centers <= max_batch_size:
            scaled_diff = scale_factor * (x.unsqueeze(0) - centers)
            dist = torch.norm(scaled_diff, p=2, dim=1).clamp(min=1e-6)
            
            if is_base_layer:
                phi = scale_factor * torch.sin(dist) / dist
                dist_shifted = dist - 0.5
                psi = scale_factor * (-torch.sin(2*dist_shifted)/dist_shifted + 
                                    torch.sin(dist_shifted)/dist_shifted)
                return phi, psi
            else:
                dist_shifted = dist - 0.5
                psi = scale_factor * (-torch.sin(2*dist_shifted)/dist_shifted + 
                                    torch.sin(dist_shifted)/dist_shifted)
                return None, psi
        else:
            if is_base_layer:
                phi_list = []
                psi_list = []
                
                for i in range(0, n_centers, max_batch_size):
                    end_idx = min(i + max_batch_size, n_centers)
                    centers_batch = centers[i:end_idx]
                    
                    scaled_diff = scale_factor * (x.unsqueeze(0) - centers_batch)
                    dist = torch.norm(scaled_diff, p=2, dim=1).clamp(min=1e-6)
                    
                    phi_batch = scale_factor * torch.sin(dist) / dist
                    dist_shifted = dist - 0.5
                    psi_batch = scale_factor * (-torch.sin(2*dist_shifted)/dist_shifted + 
                                            torch.sin(dist_shifted)/dist_shifted)
                    
                    phi_list.append(phi_batch)
                    psi_list.append(psi_batch)
                
                phi = torch.cat(phi_list)
                psi = torch.cat(psi_list)
                return phi, psi
            else:
                psi_list = []
                
                for i in range(0, n_centers, max_batch_size):
                    end_idx = min(i + max_batch_size, n_centers)
                    centers_batch = centers[i:end_idx]
                    
                    scaled_diff = scale_factor * (x.unsqueeze(0) - centers_batch)
                    dist = torch.norm(scaled_diff, p=2, dim=1).clamp(min=1e-6)
                    
                    dist_shifted = dist - 0.5
                    psi_batch = scale_factor * (-torch.sin(2*dist_shifted)/dist_shifted + 
                                            torch.sin(dist_shifted)/dist_shifted)
                    
                    psi_list.append(psi_batch)
                
                psi = torch.cat(psi_list)
                return None, psi
    
    def energy_separate(self, layer_idx, stage):
        """Energy separation for wavelet selection"""
        if layer_idx == 0:
            half = len(self.weights[layer_idx]) // 2
            weights = self.weights[layer_idx][half:]
        else:
            weights = self.init_weights.get(layer_idx)
            if weights is None:
                return torch.tensor([], device=self.device, dtype=torch.long)
        
        energy = weights * weights
        total_energy = energy.sum()
        
        if total_energy < 1e-10:
            return torch.tensor([], device=self.device, dtype=torch.long)
        
        n_elements = len(energy)
        k = min(n_elements, max(1, int(n_elements * self.mu * (stage + 1))))
        
        if stage == self.max_stage - 1:
            return torch.arange(n_elements, device=self.device, dtype=torch.long)
        
        _, top_indices = torch.topk(energy, k)
        
        sorted_energy = energy[top_indices]
        cum_energy = torch.cumsum(sorted_energy, dim=0)
        normalized_cum = cum_energy / total_energy
        
        start_ratio = stage * self.mu
        end_ratio = (stage + 1) * self.mu
        
        start_idx = torch.searchsorted(normalized_cum, start_ratio).item()
        end_idx = torch.searchsorted(normalized_cum, end_ratio).item()
        
        return top_indices[start_idx:end_idx] if end_idx > start_idx else top_indices[start_idx:start_idx+1]
    
    def _generate_child_centers_fast(self, parent_centers, new_j):
        """Fast generation of child centers"""
        new_centers = self._generate_centers(new_j)
        if len(parent_centers) == 0:
            return torch.tensor([], device=self.device, dtype=torch.long)
        
        max_pos = self.tran_r * (self.inv_scaling ** new_j)
        parent_coords = parent_centers.round()
        
        if not hasattr(self, 'offset_cache'):
            self.offset_cache = {}
        
        if self.dimension not in self.offset_cache:
            self.offset_cache[self.dimension] = torch.tensor(
                list(itertools.product([-1, 0], repeat=self.dimension)), 
                device=self.device, dtype=torch.float32
            )
        offsets = self.offset_cache[self.dimension]
        
        selected_indices = []
        max_pos_value = max_pos.item() if torch.is_tensor(max_pos) else max_pos
        
        for p_coord in parent_coords:
            candidates = (p_coord.unsqueeze(0) * 2 + offsets).clamp(0.0, max_pos_value)
            
            for candidate in candidates:
                diffs = torch.abs(new_centers - candidate).sum(dim=1)
                matches = torch.nonzero(diffs < 1e-5).view(-1)
                if len(matches) > 0:
                    selected_indices.append(matches[0].item())
        
        return torch.unique(torch.tensor(selected_indices, device=self.device, dtype=torch.long)) if selected_indices else torch.tensor([], device=self.device, dtype=torch.long)
    
    def _expand_layers(self):
        """Expand network layers"""
        current_layer = len(self.weights) - 1
        
        if self.active_stages[current_layer] < self.max_stage - 1:
            self.active_stages[current_layer] += 1
            current_stage = self.active_stages[current_layer]
            print(f"[ACWNN Stage {self.stage}] Layer {current_layer} enters stage {current_stage}")
            
            if current_layer in self.parent_stage_map:
                parent_layer = self.parent_stage_map[current_layer]['parent']
                parent_stage = self.parent_stage_map[current_layer]['stages'][current_stage]
                parent_idx = self.energy_separate(parent_layer, parent_stage)
                
                if len(parent_idx) > 0:
                    new_j = self.j + current_layer
                    parent_centers = self.centers_list[parent_layer][parent_idx]
                    selected_idx = self._generate_child_centers_fast(parent_centers, new_j)
                    
                    if len(selected_idx) > 0:
                        combined_idx = torch.cat([self.active_indices[current_layer], selected_idx])
                        self.active_indices[current_layer] = torch.unique(combined_idx)
                        print(f"  Added {len(selected_idx)} new bases, total active: {len(self.active_indices[current_layer])}")
                
                if current_stage == self.max_stage - 1 and current_layer > 0:
                    total = len(self.centers_list[current_layer])
                    self.active_indices[current_layer] = torch.arange(total, device=self.device)
                    print(f"[ACWNN Stage {self.stage}] Layer {current_layer} fully activated")
        else:
            self._add_new_layer(current_layer)
    
    def _add_new_layer(self, parent_layer):
        """Add new network layer"""
        new_layer_idx = len(self.weights)
        new_j = self.j + new_layer_idx
        new_centers = self._generate_centers(new_j)
        
        self.init_weights[parent_layer] = self.weights[parent_layer].data.clone()
        
        self.parent_stage_map[new_layer_idx] = {
            'parent': parent_layer, 
            'stages': {i: i for i in range(self.max_stage)}
        }
        
        if self.mu == 1:
            selected_idx = torch.arange(len(new_centers), device=self.device)
        else:
            parent_idx = self.energy_separate(parent_layer, 0)
            print(f"  Energy separate: selected {len(parent_idx)} parent bases from layer {parent_layer}")
            if len(parent_idx) > 0:
                parent_centers = self.centers_list[parent_layer][parent_idx]
                selected_idx = self._generate_child_centers_fast(parent_centers, new_j)
                print(f"  Generated {len(selected_idx)} child centers for new layer {new_layer_idx}")
            else:
                selected_idx = torch.tensor([], device=self.device, dtype=torch.long)
        
        self.centers_list.append(new_centers)
        self.active_indices.append(selected_idx)
        
        if self.mu == 1:
            self.active_stages.append(self.max_stage - 1)
        else:
            self.active_stages.append(0)
        
        param_size = len(new_centers)
        new_weight = nn.Parameter(torch.zeros(param_size, device=self.device))
        self.weights.append(new_weight)
        self.init_weights[new_layer_idx] = torch.zeros(param_size, device=self.device)
        
        self.adam_m.append(torch.zeros(param_size, device=self.device))
        self.adam_v.append(torch.zeros(param_size, device=self.device))
        
        print(f"[ACWNN Stage {self.stage}] Added new layer {new_layer_idx} (j={new_j}), active bases: {len(selected_idx)}")
    
    def _check_dwell_time(self, t):
        """Check if a dwell time period is completed (no pruning version)"""
        current_time = t.item()
        
        if current_time >= self.current_dwell_start + self.dwell_time - 1e-6:
            if self.dwell_losses:
                max_error = max(self.dwell_losses)
                avg_error = np.mean(self.dwell_losses)
                min_error = min(self.dwell_losses)
                
                # Save statistics
                if not hasattr(self, 'dwell_statistics'):
                    self.dwell_statistics = []
                
                self.dwell_statistics.append({
                    'start_time': self.current_dwell_start,
                    'end_time': current_time,
                    'dwell_time': self.dwell_time,
                    'max_error': max_error,
                    'avg_error': avg_error,
                    'min_error': min_error,
                    'sample_count': len(self.dwell_losses),
                    'stage': self.stage,
                    'N_counter': self.N_counter
                })
                
                print(f"\n[ACWNN Stage {self.stage}] Dwell time completed at t={current_time:.2f}")
                print(f"  ΔT={self.dwell_time:.2f}, max_error={max_error:.6f}, avg_error={avg_error:.6f}")
                
                # Memory report
                current_rss = self.process.memory_info().rss / 1024**3
                rss_increase = current_rss - self.initial_rss
                print(f"  [Memory at dwell end] RSS: {current_rss:.2f}GB (increase: {rss_increase:.2f}GB)")
                
                # Garbage collection
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                # N_d functionality (no pruning version)
                if max_error <= self.delta_ac:
                    self.N_counter += 1
                    self.dwell_time *= 2
                    print(f"  ✓ Error satisfied! N_counter={self.N_counter}/{self.N_d}, ΔT→{self.dwell_time:.2f}")
                    
                    if self.N_counter >= self.N_d:
                        print(f"\n{'='*50}")
                        print(f"ACWNN: Reached N_d={self.N_d}! System stabilized!")
                        print(f"{'='*50}\n")
                else:
                    print(f"  ✗ Error not satisfied, expanding layers...")
                    self._expand_layers()
                    # Reset on failure
                    self.N_counter = 0
                    self.dwell_time = self.config['base_interval']
                    print(f"  Reset: N_counter→0, ΔT→{self.dwell_time:.2f}")
            
            self.current_dwell_start = current_time
            self.dwell_losses.clear()
    
    @torch.jit.unused
    def _compute_wavelet_sum(self, wavelet_cache):
        """Compute weighted sum of wavelet functions"""
        wavelet_sum = torch.tensor(0.0, device=self.device)
        
        for layer_idx, cache in wavelet_cache.items():
            if layer_idx == 0:
                weights = self.weights[0]
                combined = cache['combined']
                wavelet_sum += torch.dot(weights, combined)
            else:
                active_idx = self.active_indices[layer_idx]
                if len(active_idx) > 0:
                    weights = self.weights[layer_idx][active_idx]
                    psi = cache['psi']
                    wavelet_sum += torch.dot(weights, psi)
        
        return wavelet_sum
    
    def _update_all_parameters(self, wavelet_cache, deta):
        """Batch update all parameters using ADAM optimizer"""
        self.adam_t += 1
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        bias_correction1 = 1 - beta1**self.adam_t
        bias_correction2 = 1 - beta2**self.adam_t
        lr_adjusted = self.learning_rate * self.inv_scaling / bias_correction1
        
        for layer_idx, cache in wavelet_cache.items():
            if layer_idx == 0:
                delta = cache['combined'] * deta
                self._adam_update_acwnn(0, delta, beta1, beta2, eps, 
                                    bias_correction2, lr_adjusted)
            else:
                active_idx = self.active_indices[layer_idx]
                if len(active_idx) > 0:
                    j = cache['j']
                    scale = self.power_cache.get(j, self.inv_scaling ** j)
                    delta = cache['psi'] * deta * scale
                    self._adam_update_acwnn(layer_idx, delta, beta1, beta2, eps,
                                        bias_correction2, lr_adjusted, active_idx)
    
    def _adam_update_acwnn(self, layer_idx, delta, beta1, beta2, eps,
                        bias_correction2, lr_adjusted, active_idx=None):
        """ADAM update for no-pruning version"""
        w = self.weights[layer_idx]
        m = self.adam_m[layer_idx]
        v = self.adam_v[layer_idx]
        
        beta1_comp = 1 - beta1
        beta2_comp = 1 - beta2
        
        if layer_idx == 0:
            m_new = beta1 * m + beta1_comp * delta
            v_new = beta2 * v + beta2_comp * (delta * delta)
            
            m[:] = m_new
            v[:] = v_new
            
            v_hat = v_new / bias_correction2
            update = lr_adjusted * m_new / (torch.sqrt(v_hat) + eps)
            w.data = w.data + update
        else:
            if active_idx is not None and len(active_idx) > 0:
                m_old = m[active_idx]
                v_old = v[active_idx]
                
                m_new = beta1 * m_old + beta1_comp * delta
                v_new = beta2 * v_old + beta2_comp * (delta * delta)
                
                m[active_idx] = m_new
                v[active_idx] = v_new
                
                v_hat = v_new / bias_correction2
                update = lr_adjusted * m_new / (torch.sqrt(v_hat) + eps)
                w.data[active_idx] = w.data[active_idx] + update

    def forward(self, t, state):
        """Forward pass for no-pruning ACWNN"""
        if self.stop_flag:
            return torch.zeros_like(state)
        
        state = state.detach()
        
        x_est = state[:self.system_order]
        
        traj = self.trajectory_processor.compute_derivatives(t)
        
        errors = []
        errors.append(x_est[0] - traj['xr'])
        for i in range(1, self.system_order):
            errors.append(x_est[i] - traj[f'xr_dot{i}'])
        
        deta = sum(coeff * error for coeff, error in zip(self.control_coeffs, errors))
        
        deta_abs = abs(deta.item())
        self.dwell_losses.append(deta_abs)
        
        self._check_dwell_time(t)
        
        wavelet_cache = self._compute_wavelet_terms_fast(x_est)
        wavelet_sum = self._compute_wavelet_sum(wavelet_cache)
        
        # Compute nonlinear term and add disturbance if enabled
        current_time = t.item()
        nonlinear_term = self.compute_nonlinear(x_est)
        
        # Apply disturbance in specified time window
        if self.disturbance_enabled:
            if self.disturbance_start_time <= current_time <= self.disturbance_end_time:
                disturbance = self.disturbance_magnitude * torch.sin(4 * t)
                nonlinear_term = nonlinear_term + disturbance
        
        velocity_feedback = sum(self.control_coeffs[i] * errors[i+1] 
                               for i in range(self.system_order - 1))
        velocity_term = traj[f'xr_dot{self.system_order}'] - velocity_feedback
        
        acceleration = -self.beta * deta - wavelet_sum + nonlinear_term + velocity_term
        
        self._update_all_parameters(wavelet_cache, deta)
        
        X = torch.zeros_like(state)
        for i in range(self.system_order - 1):
            X[i] = x_est[i + 1]
        X[self.system_order - 1] = acceleration
        
        self._periodic_logging(t, deta)
        
        return X.detach()
    
    def _periodic_logging(self, t, deta):
        """Periodic logging and printing"""
        current_time = t.item()
        if current_time - self.last_print_time >= 0.05:
            # Count all parameters
            param_count = len(self.weights[0])
            
            for layer_idx in range(1, len(self.weights)):
                active_idx = self.active_indices[layer_idx]
                param_count += len(active_idx)
            
            # Memory monitoring
            current_rss = self.process.memory_info().rss / 1024**3
            rss_increase = current_rss - self.initial_rss
            
            self.memory_check_points.append({
                'time': current_time,
                'rss_gb': current_rss,
                'increase_gb': rss_increase
            })
            
            records = self.training_records
            records['time'].append(current_time)
            records['loss'].append(deta.item())
            records['params_count'].append(param_count)
            records['stage'].append(self.stage)
            records['dwell_time'].append(self.dwell_time)
            records['N_counter'].append(self.N_counter)
            
            print(f"[t={current_time:.2f}] ACWNN Stage={self.stage}, loss={deta.item():.6f}, "
                f"layers={len(self.weights)}, params={param_count}, "
                f"N_counter={self.N_counter}/{self.N_d}, ΔT={self.dwell_time:.2f}, "
                f"RSS={current_rss:.2f}GB (+{rss_increase:.2f}GB)")
            
            if int(current_time) % 10 == 0 and current_time > 0:
                print(f"  [Memory Report] RSS: {current_rss:.2f}GB, Increase: {rss_increase:.2f}GB")
                if self.device == 'cuda':
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    print(f"  [GPU Memory] {gpu_mem:.2f}GB")
            
            self.last_print_time = current_time
    
    def calculate_window_statistics(self, df):
        """Calculate statistics for each window"""
        window_stats = []
        
        if hasattr(self, 'dwell_statistics') and self.dwell_statistics:
            for i, dwell_stat in enumerate(self.dwell_statistics):
                window_data = df[(df['time'] >= dwell_stat['start_time']) & 
                               (df['time'] <= dwell_stat['end_time'])]
                
                stats = {
                    'Window Number': i + 1,
                    'Start Time (s)': dwell_stat['start_time'],
                    'End Time (s)': dwell_stat['end_time'],
                    'Dwell Time': dwell_stat['dwell_time'],
                    'Mean |δ|': dwell_stat['avg_error'],
                    'Max |δ|': dwell_stat['max_error'],
                    'Min |δ|': dwell_stat['min_error'],
                    'Stage': dwell_stat['stage'],
                    'N_counter': dwell_stat['N_counter'],
                    'Data Points (full)': dwell_stat['sample_count'],
                    'Data Points (logged)': len(window_data)
                }
                
                if len(window_data) > 0:
                    stats['Max δ (with sign)'] = window_data['loss'].max()
                    stats['Min δ (with sign)'] = window_data['loss'].min()
                    if 'params_count' in window_data.columns:
                        stats['Final Params'] = window_data['params_count'].iloc[-1]
                
                window_stats.append(stats)
        
        return window_stats

    def save_to_excel(self, filename=None):
        """Save training data to Excel with automatic filename generation"""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mu_str = f"{self.mu:.2f}".replace('.', 'p')
            
            disturbance_str = "_disturbed" if self.disturbance_enabled else ""
            
            filename = (f"training_report_ACWNN_nopruning_{self.system_order}D_"
                       f"mu{mu_str}_dt{self.config['base_interval']}"
                       f"{disturbance_str}.xlsx")
        
        df = pd.DataFrame(dict(self.training_records))
        df.sort_values(by='time', inplace=True)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Training Data', index=False)
            
            config_df = pd.DataFrame({
                'Parameter': ['system_order', 'lambda_c', 'beta', 'j_init', 'mu',
                            'N_d', 'error_threshold', 'initial_dwell_time'],
                'Value': [self.system_order, self.lambda_c, self.beta,
                        self.j, self.mu, self.N_d, self.delta_ac,
                        self.config['base_interval']]
            })
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
            
            window_stats = self.calculate_window_statistics(df)
            if window_stats:
                window_df = pd.DataFrame(window_stats)
                window_df.to_excel(writer, sheet_name='Window Statistics', index=False)
                print(f"  - Window statistics saved ({len(window_stats)} windows)")
                
                max_errors = [w['Max |δ|'] for w in window_stats]
                print(f"  - Max errors across windows: min={min(max_errors):.6f}, max={max(max_errors):.6f}")
                
                mean_errors = [w['Mean |δ|'] for w in window_stats]
                best_window = mean_errors.index(min(mean_errors)) + 1
                print(f"  - Best window: Window {best_window} (Mean |δ|={min(mean_errors):.6f})")
        
        print(f"Training data saved to {filename}")
        return filename

    def plot_training_progress(self):
        """Plot training progress with automatic filename generation"""
        if not self.training_records['time']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        data = dict(self.training_records)
        
        axes[0, 0].plot(data['time'], data['loss'])
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss (ACWNN - No Pruning)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].plot(data['time'], data['params_count'])
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Number of Parameters')
        axes[0, 1].set_title('Active Parameters (ACWNN - No Pruning)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(data['time'], data['stage'])
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Stage')
        axes[1, 0].set_title('Algorithm Stage (ACWNN - Always Stage 1)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0.5, 1.5])
        
        axes[1, 1].plot(data['time'], data['dwell_time'])
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Dwell Time (ΔT)')
        axes[1, 1].set_title('Adaptive Dwell Time (ACWNN - No Pruning)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mu_str = f"{self.mu:.2f}".replace('.', 'p')
        
        disturbance_str = "_disturbed" if self.disturbance_enabled else ""
        
        plot_path = os.path.join(self.plot_dir, 
                                f"training_progress_ACWNN_nopruning_{self.system_order}D_mu{mu_str}_"
                                f"dt{self.config['base_interval']}{disturbance_str}_{timestamp}.png")
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training progress plot saved to {plot_path}")
        return plot_path


def plot_tracking_error_standalone(training_records, config=None, save_path=None):
    """Standalone tracking error plotting function"""
    
    if isinstance(training_records, defaultdict):
        df = pd.DataFrame(dict(training_records))
    else:
        df = pd.DataFrame(training_records)
    df.sort_values(by='time', inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t = df['time'].values
    tracking_error = df['loss'].values
    ax.plot(t, tracking_error, 'b-', linewidth=1.5, label='ACWNN (No Pruning)')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Augmented tracking error', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    if 'dwell_time' in df.columns:
        dwell_changes = df[['time', 'dwell_time']].drop_duplicates(subset=['dwell_time'])
        
        window_boundaries = []
        current_time = 0
        
        for idx, row in dwell_changes.iterrows():
            dwell_duration = row['dwell_time']
            while current_time < t[-1]:
                current_time += dwell_duration
                if current_time <= t[-1]:
                    window_boundaries.append(current_time)
                    next_change = df[df['time'] >= current_time]['dwell_time'].iloc[0] if len(df[df['time'] >= current_time]) > 0 else dwell_duration
                    if next_change != dwell_duration:
                        dwell_duration = next_change
                        break
        
        for boundary in window_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    else:
        base_interval = config.get('base_interval', 150) if config else 150
        time_marks = np.arange(base_interval, t[-1] + 1, base_interval)
        for tm in time_marks:
            ax.axvline(x=tm, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    
    if config:
        target = config.get('error_threshold', 0.1)
        ax.axhline(y=target, color='r', linestyle='--', linewidth=1.0, 
                   alpha=0.8, label=f'Target ({target})')
        ax.axhline(y=-target, color='r', linestyle='--', linewidth=1.0, alpha=0.8)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(config.get('base_interval', 150)))
    ax.set_xlim([0, t[-1]])
    
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config:
            system_order = config.get('system_order', 4)
            mu_value = config.get('mu', 1/3)
            base_interval = config.get('base_interval', 150)
            lambda_c = config.get('lambda_c', 2.0)
            beta = config.get('beta', 10.0)
            
            disturbance_enabled = config.get('disturbance_enabled', False)
            disturbance_str = "_disturbed" if disturbance_enabled else ""
            
            mu_str = f"{mu_value:.2f}".replace('.', 'p')
            save_path = (f'tracking_error_ACWNN_nopruning_{system_order}D_'
                        f'mu{mu_str}_dt{base_interval}_'
                        f'lambda{lambda_c}_beta{beta}'
                        f'{disturbance_str}.pdf')
        else:
            save_path = f'tracking_error_ACWNN_nopruning.pdf'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Tracking error plot saved as: {save_path}")
    
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration for 4D system
    config_four_dim = {
        'j_init': 1,
        'mu': 1,
        'tran_r': 1,
        'base_interval': 120,
        'time_max': 961,
        'max_layers': 5,

        # Cache control
        'max_cache_size': 10000,
        'max_cacheable_j': 5,
        
        # System parameters
        'system_order': 4,
        'lambda_c': 2.0,
        'beta': 10.0,
        
        'trajectory_function': None,
        'nonlinear_function': None,
        
        'N_d': 1,
        'error_threshold': 0.12,  # Single threshold (no pruning)
        
        # Disturbance configuration (optional feature)
        'disturbance_enabled': True,
        'disturbance_start_time': 120.0,
        'disturbance_end_time': 24000.0,
        'disturbance_magnitude': 6,
    }
    
    # Configuration for 2D system
    def custom_trajectory(t):
        """Custom trajectory function"""
        return 0.5 * torch.sin(t) ** 2
    
    def custom_nonlinear(x):
        """Custom nonlinear function"""
        return 8*torch.exp(-x[0]**2) * torch.sin(5*torch.sqrt(x[0]**2 + x[1]**2)) + torch.sin(4*x[1])
    
    config_two_dim = {
        'j_init': 1,
        'mu': 1,
        'tran_r': 2,
        'base_interval': 120,
        'time_max': 961,
        'max_layers': 5,

        # Cache control
        'max_cache_size': 10000,
        'max_cacheable_j': 5,
        
        'system_order': 2,
        'lambda_c': 20.0,
        'beta': 20.0,
        
        'trajectory_function': custom_trajectory,
        'nonlinear_function': custom_nonlinear,
        
        'N_d': 1,
        'error_threshold': 0.04,
    }
    
    # Select configuration
    config = config_four_dim  # Use 4D system
    #config = config_two_dim  # Or use 2D system
    
    # Create model
    model = AdaptiveWaveletControl(config, device)
    
    # Initial state
    initial_state = torch.zeros(config['system_order'] + 1, device=device)
    t_span = torch.linspace(0, config['time_max'], 3*config['time_max']).to(device)
    
    print("\n" + "="*50)
    print("Starting ACWNN Training (No Pruning Version)")
    print(f"System Order: {config['system_order']}")
    print(f"Lambda: {config['lambda_c']}")
    print(f"Control Coefficients: {model.control_coeffs}")
    if config.get('disturbance_enabled', False):
        print(f"Disturbance: Enabled from t={config['disturbance_start_time']} to t={config['disturbance_end_time']}")
    print("="*50 + "\n")
    
    # Run simulation
    states = odeint(
        model, 
        initial_state, 
        t_span, 
        method='dopri5',
        rtol=1e-6,   
        atol=1e-7,
        options={'min_step': 1e-7}
    )
    
    print("\n" + "="*50)
    print("ACWNN Training completed!")

    # Memory usage summary
    final_rss = model.process.memory_info().rss / 1024**3
    total_increase = final_rss - model.initial_rss
    print(f"[Memory Summary]")
    print(f"  Initial RSS: {model.initial_rss:.2f}GB")
    print(f"  Final RSS: {final_rss:.2f}GB")
    print(f"  Total Increase: {total_increase:.2f}GB")

    print("="*50 + "\n")
    
    # Save Excel
    excel_filename = model.save_to_excel()
    
    # Plot training progress
    progress_plot = model.plot_training_progress()
    
    # Plot tracking error
    fig, ax = plot_tracking_error_standalone(
        model.training_records, 
        config,
        save_path=None
    )
    
    print("\n" + "="*50)
    print("All results saved:")
    print(f"  - Excel: {excel_filename}")
    print(f"  - Progress plot: {progress_plot}")
    print("="*50)