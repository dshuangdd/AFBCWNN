"""
Simplified Initial Resolution Estimator for AFBCWNN

Key modifications:
1. EMA calculation uses each layer's own update count
2. Added 4D system configuration support

Author: Based on ACWNN framework
Date: 2025
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import time
from datetime import datetime
import math


class AutomaticTrajectoryProcessor:
    """Automatic processor for trajectory function and its derivatives"""
    
    def __init__(self, trajectory_func, system_order, device):
        self.trajectory_func = trajectory_func
        self.system_order = system_order
        self.device = device
        self.derivatives_cache = {}
        self.use_manual = False
        self.max_cache_size = 10
        
    def set_manual_derivatives(self, derivatives_func):
        """Set manually computed derivatives function"""
        self.manual_derivatives = derivatives_func
        self.use_manual = True
        
    def compute_derivatives(self, t):
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


class SimplifiedResolutionEstimator(nn.Module):
    """Simplified initial resolution estimator for AFBCWNN"""
    
    def __init__(self, config, device):
        super().__init__()
        
        self.config = config
        self.device = device
        
        self.system_order = config['system_order']
        self.dimension = self.system_order
        self.tran_r = config['tran_r']
        self.dwell_time = config['dwell_time']
        
        self.lambda_c = config['lambda_c']
        self.beta = config['beta']
        
        self.scaling = 0.5
        self.inv_scaling = 2.0
        self.learning_rate = config['lr']
        
        self._setup_trajectory_processor(config)
        self._setup_nonlinear_function(config)
        self._compute_control_coefficients()
        
        self.max_cache_size = config.get('max_cache_size', 10000)
        self.max_cacheable_j = config.get('max_cacheable_j', 5)
        
        # Pre-compute power cache
        self.power_cache = {}
        for j in range(10):
            self.power_cache[j] = self.inv_scaling ** j
        
        self.offset_cache = {}
        self.grid_cache = {}
        self.mesh_cache = {}
        self._precompute_grid_templates()
        
        self.current_resolution = None
        self.all_layers = []
        self.adam_states = []
        
        self.wavelet_cache = {}
        self.cache_x_est = None
        
        self.losses = []
        
    def _compute_control_coefficients(self):
        """Compute control coefficients based on system order and lambda"""
        d = self.system_order
        λ = self.lambda_c
        
        self.control_coeffs = []
        
        for k in range(d):
            n = d - 1
            comb_value = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
            coeff = comb_value * (λ ** (d - 1 - k))
            self.control_coeffs.append(coeff)
        
        print(f"  System order: {d}, λ: {λ}")
        print(f"  Control coefficients: {self.control_coeffs}")
    
    def _setup_trajectory_processor(self, config):
        """Setup trajectory processor with default or custom trajectory"""
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
        """Setup nonlinear function with default or custom implementation"""
        nonlinear_func = config.get('nonlinear_function')
        
        if nonlinear_func is None:
            self.compute_nonlinear = self._original_nonlinear
        else:
            self.compute_nonlinear = nonlinear_func
    
    def _original_nonlinear(self, x_est):
        """Default nonlinear function for the system"""
        if self.system_order >= 4:
            x0, x1, x2, x3 = x_est[0], x_est[1], x_est[2], x_est[3]
            x_sum = x0 + 2*(x1 + x2 + x3)
            x_partial = x0 + 2*x1 + x2
            x_tail = x1 + 2*x2 + x3
            return -x_sum + (1 - x_partial*x_partial) * x_tail
        else:
            return -torch.sum(x_est[:self.system_order])
    
    def _precompute_grid_templates(self):
        """Pre-compute grid templates for common resolutions"""
        max_j_to_cache = min(10, self.max_cacheable_j)
        
        for j in range(max_j_to_cache):
            grid_size = self.tran_r * 2**j + 1
            
            if grid_size <= 100:
                grid = torch.arange(grid_size, device=self.device, dtype=torch.float32)
                self.grid_cache[grid_size] = grid
                print(f"Pre-cached grid for j={j}, grid_size={grid_size}")
    
    def _generate_centers(self, j):
        """Generate wavelet centers for resolution j"""
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
    
    def _equidistant_selection(self, all_centers, N_J, j):
        """Equidistant sampling for center selection"""
        grid_size = self.tran_r * 2**j + 1
        dimension = self.dimension
        
        N_l = max(1, int(np.round(N_J ** (1.0 / dimension))))
        while N_l > 1 and (N_l ** dimension) > N_J * 1.5:
            N_l -= 1
        
        dim_indices = []
        for d in range(dimension):
            if N_l > 1:
                indices = np.linspace(0, grid_size - 1, N_l, dtype=int)
            else:
                indices = np.array([grid_size // 2], dtype=int)
            dim_indices.append(indices)
        
        index_combinations = list(itertools.product(*dim_indices))
        
        selected_flat_indices = []
        for combo in index_combinations:
            flat_idx = 0
            for d, idx in enumerate(combo):
                flat_idx = flat_idx * grid_size + idx
            selected_flat_indices.append(flat_idx)
        
        selected_flat_indices = torch.tensor(selected_flat_indices, 
                                            device=self.device, 
                                            dtype=torch.long)
        
        selected_flat_indices = selected_flat_indices[selected_flat_indices < len(all_centers)]
        selected_flat_indices = torch.unique(selected_flat_indices)
        
        print(f"    Equidistant selection: {len(selected_flat_indices)} centers")
        
        return selected_flat_indices
    
    def _generate_child_centers_fast(self, parent_centers, new_j):
        """Fast generation of child centers from parent centers"""
        new_centers = self._generate_centers(new_j)
        
        if len(parent_centers) == 0:
            return torch.tensor([], device=self.device, dtype=torch.long)
        
        max_pos = self.tran_r * (self.inv_scaling ** new_j)
        parent_coords = parent_centers.round()
        
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
        
        if not selected_indices:
            return torch.tensor([], device=self.device, dtype=torch.long)
        
        unique_indices = torch.unique(torch.tensor(selected_indices, 
                                                   device=self.device, 
                                                   dtype=torch.long))
        
        print(f"    Generated {len(unique_indices)} child centers from "
              f"{len(parent_centers)} parents")
        
        return unique_indices
    
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
            
            for layer_idx in range(len(self.all_layers)):
                layer = self.all_layers[layer_idx]
                j = layer['resolution']
                centers = layer['centers']
                
                psi = self._wavelet_function_batch(x_est, centers, j)
                
                self.wavelet_cache[layer_idx] = {'psi': psi, 'j': j}
        
        return self.wavelet_cache
    
    def _wavelet_function_batch(self, x, centers, j):
        """Batch computation of wavelet function"""
        scale_factor = self.power_cache.get(j, self.inv_scaling ** j)
        
        n_centers = len(centers)
        max_batch_size = 5000
        
        if n_centers <= max_batch_size:
            scaled_diff = scale_factor * (x.unsqueeze(0) - centers)
            dist = torch.norm(scaled_diff, p=2, dim=1).clamp(min=1e-6)
            
            dist_shifted = dist - 0.5
            psi = scale_factor * (-torch.sin(2*dist_shifted)/dist_shifted + 
                                torch.sin(dist_shifted)/dist_shifted)
            return psi
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
            return psi
    
    def _compute_wavelet_sum(self, wavelet_cache):
        """Compute weighted sum of wavelet functions"""
        wavelet_sum = torch.tensor(0.0, device=self.device)
        
        for layer_idx, cache in wavelet_cache.items():
            layer = self.all_layers[layer_idx]
            weights = layer['weights']
            psi = cache['psi']
            
            wavelet_sum += torch.dot(weights, psi)
        
        return wavelet_sum
    
    def _adam_update_high_precision(self, layer_idx, delta, grad):
        """High-precision ADAM parameter update"""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        beta1_comp = 1 - beta1
        beta2_comp = 1 - beta2
        
        layer = self.all_layers[layer_idx]
        weights = layer['weights']
        
        adam_state = self.adam_states[layer_idx]
        adam_state['t'] += 1
        t = adam_state['t']
        
        m = adam_state['m']
        v = adam_state['v']
        
        m_new = beta1 * m + beta1_comp * grad
        v_new = beta2 * v + beta2_comp * (grad * grad)
        
        adam_state['m'] = m_new
        adam_state['v'] = v_new
        
        bias_correction1 = 1 - beta1**t
        bias_correction2 = 1 - beta2**t
        lr_adjusted = self.learning_rate * self.inv_scaling / bias_correction1
        
        v_hat = v_new / bias_correction2
        update = lr_adjusted * m_new / (torch.sqrt(v_hat) + eps)
        weights.data = weights.data - update
    
    def _update_all_parameters(self, wavelet_cache, delta):
        """Update parameters for all layers"""
        for layer_idx, cache in wavelet_cache.items():
            layer = self.all_layers[layer_idx]
            j = cache['j']
            psi = cache['psi']
            
            scale = self.power_cache.get(j, self.inv_scaling ** j)
            grad = psi * delta * scale
            
            self._adam_update_high_precision(layer_idx, delta, grad)
    
    def forward(self, t, state):
        """Forward propagation for ODE integration"""
        state = state.detach()
        x_est = state[:self.system_order]
        
        traj = self.trajectory_processor.compute_derivatives(t)
        errors = []
        errors.append(x_est[0] - traj['xr'])
        for i in range(1, self.system_order):
            errors.append(x_est[i] - traj[f'xr_dot{i}'])
        
        delta = sum(coeff * error for coeff, error in zip(self.control_coeffs, errors))
        
        deta_abs = abs(delta.item())
        self.losses.append(deta_abs)
        
        wavelet_cache = self._compute_wavelet_terms_fast(x_est)
        wavelet_sum = self._compute_wavelet_sum(wavelet_cache)
        
        nonlinear_term = self.compute_nonlinear(x_est)
        
        velocity_feedback = sum(self.control_coeffs[i] * errors[i+1] 
                               for i in range(self.system_order - 1))
        velocity_term = traj[f'xr_dot{self.system_order}'] - velocity_feedback
        
        acceleration = -self.beta * delta - wavelet_sum + nonlinear_term + velocity_term
        
        self._update_all_parameters(wavelet_cache, delta)
        
        X = torch.zeros_like(state)
        for i in range(self.system_order - 1):
            X[i] = x_est[i + 1]
        X[self.system_order - 1] = acceleration
        
        return X.detach()
    
    def _initialize_layer(self, resolution, centers, total_centers_in_space):
        """Initialize a new layer with given centers"""
        n_centers = len(centers)
        weights = nn.Parameter(torch.zeros(n_centers, device=self.device))
        
        layer = {
            'resolution': resolution,
            'weights': weights,
            'centers': centers,
            'n_selected': n_centers,
            'n_total': total_centers_in_space
        }
        
        self.all_layers.append(layer)
        
        layer_idx = len(self.all_layers) - 1
        self.register_parameter(f'layer_{layer_idx}_weights', weights)
        
        self.adam_states.append({
            'm': torch.zeros(n_centers, device=self.device),
            'v': torch.zeros(n_centers, device=self.device),
            't': 0
        })
        
        print(f"    Initialized layer {layer_idx}: m={resolution}, "
              f"{n_centers} selected / {total_centers_in_space} total")
    
    def _run_one_resolution(self, initial_state, t_start, t_end):
        """Run training for one resolution"""
        print(f"    Running ODE: t=[{t_start:.1f}, {t_end:.1f}]")
        
        self.losses = []
        
        total_time = t_end - t_start
        n_points = int(total_time * 3)
        t_span = torch.linspace(t_start, t_end, n_points, device=self.device)
        
        print(f"      Evaluation points: {n_points} (3 points/sec)")
        
        start_time = time.time()
        try:
            states = odeint(
                self,
                initial_state,
                t_span,
                method='dopri5',
                rtol=1e-6,
                atol=1e-7,
                options={'min_step': 1e-7}
            )
            elapsed = time.time() - start_time
            
            print(f"      Completed in {elapsed:.2f}s")
            print(f"      Forward called {len(self.losses)} times")
            
        except Exception as e:
            print(f"      ODE integration failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
        
        final_state = states[-1].clone()
        
        max_loss = max(self.losses) if self.losses else 0.0
        mean_loss = np.mean(self.losses) if self.losses else 0.0
        
        return final_state, max_loss, mean_loss
    
    def estimate_m_star(self, initial_state, max_resolution=5):
        """
        Estimate optimal initial resolution m*
        
        Algorithm logic:
        1. After each iteration, recompute E_hat_m for all layers using latest weights
        2. Recompute all E_bar_m in resolution order
        3. Peak detection: E_bar_{m-1} >= E_hat_m
        """
        print("\n" + "="*70)
        print(f"Simplified Initial Resolution Estimation (Corrected Algorithm)")
        print("="*70)
        
        # Step 1: Initialize W1 with equidistant sampling
        print("\n[Step 1: Initialize W1 with equidistant sampling]")
        j_init = 1
        all_centers_1 = self._generate_centers(j_init)
        N_1 = len(all_centers_1)
        
        sampling_ratio = self.config.get('sampling_ratio', 0.36)
        N_J = int(np.floor(sampling_ratio * N_1))
        
        print(f"  W_1 total bases: {N_1}")
        print(f"  Sampling ratio: {sampling_ratio}")
        print(f"  Selected bases: {N_J}")
        
        selected_indices_1 = self._equidistant_selection(all_centers_1, N_J, j_init)
        selected_centers_1 = all_centers_1[selected_indices_1]
        
        # Step 2: Iterative search for m* using energy peak detection
        print("\n[Step 2: Iterative search for m* using energy peak detection]")
        
        alpha = self._compute_ema_alpha()
        print(f"  EMA coefficient alpha: {alpha:.4f} (from error_threshold={self.config.get('error_threshold', 0.1)})")
        print(f"  Dwell time per resolution: {self.dwell_time}s")
        print(f"\n  Algorithm (from pseudocode):")
        print(f"     if m = 1: E_bar_1 = E_hat_1")
        print(f"     else: E_bar_m = (alpha*E_bar_{{m-1}} + (1-alpha)*E_hat_m) / (1-alpha^m)")
        print(f"     Peak detection: E_bar_{{m-1}} >= E_hat_m")
        
        # Organize energy by resolution (not by layer)
        E_hat_by_resolution = {}  # {m: E_hat_m} - latest energy
        E_bar_by_resolution = {}  # {m: E_bar_m} - EMA
        
        # Record history for visualization
        energy_history = {
            'resolution': [],
            'max_loss': [],
            'mean_loss': [],
            'n_total_bases': [],
            'odeint_time': [],
            't_start': [],
            't_end': [],
            'E_hat_history': defaultdict(list),  # {m: [E_hat_m at each iteration]}
            'E_bar_history': defaultdict(list),  # {m: [E_bar_m at each iteration]}
        }
        
        parent_centers_for_next = selected_centers_1
        current_state = initial_state.clone()
        
        m = 1
        m_star = None
        iteration = 0
        
        while m_star is None and m <= max_resolution:
            iteration += 1
            print(f"\n  {'='*66}")
            print(f"  Iteration {iteration}: Adding Resolution m={m}")
            print(f"  {'='*66}")
            
            t_start = (m - 1) * self.dwell_time
            t_end = m * self.dwell_time
            
            # Get current layer centers
            if m == 1:
                current_centers = selected_centers_1
                all_centers_m = all_centers_1
            else:
                all_centers_m = self._generate_centers(m)
                selected_indices_m = self._generate_child_centers_fast(
                    parent_centers_for_next, m
                )
                
                if len(selected_indices_m) == 0:
                    print(f"    Warning: No child centers found, stopping")
                    break
                
                current_centers = all_centers_m[selected_indices_m]
            
            # Add new layer
            total_centers_in_space = len(all_centers_m)
            self._initialize_layer(m, current_centers, total_centers_in_space)
            
            n_total = sum(len(layer['weights']) for layer in self.all_layers)
            
            energy_history['n_total_bases'].append(n_total)
            energy_history['t_start'].append(t_start)
            energy_history['t_end'].append(t_end)
            
            # Run ODE training
            odeint_start = time.time()
            final_state, max_loss, mean_loss = self._run_one_resolution(
                current_state, t_start, t_end
            )
            odeint_time = time.time() - odeint_start
            energy_history['odeint_time'].append(odeint_time)
            
            if final_state is None:
                print(f"    Failed to run resolution m={m}")
                break
            
            current_state = final_state
            
            print(f"    Max |delta|: {max_loss:.6f}, Mean |delta|: {mean_loss:.6f}")
            
            energy_history['max_loss'].append(max_loss)
            energy_history['mean_loss'].append(mean_loss)
            energy_history['resolution'].append(m)
            
            # Step 1: Recompute E_hat_m for all resolutions with latest weights
            print(f"\n    Step 1: Computing E_hat_m for all resolutions with latest weights:")
            
            for layer_idx, layer in enumerate(self.all_layers):
                layer_resolution = layer['resolution']
                
                # Get latest weights
                layer_weights = layer['weights'].data
                E_hat_raw = torch.sum(layer_weights ** 2).item()
                
                # Use raw energy without correction
                E_hat = E_hat_raw
                
                # Store by resolution (will overwrite old values)
                E_hat_by_resolution[layer_resolution] = E_hat
                
                print(f"      m={layer_resolution} (Layer {layer_idx}): "
                    f"E_hat={E_hat:.6f}")
            
            # Step 2: Recompute all E_bar_m in resolution order
            print(f"\n    Step 2: Computing E_bar_m for all resolutions (EMA):")
            
            for res in sorted(E_hat_by_resolution.keys()):
                E_hat_curr = E_hat_by_resolution[res]
                
                if res == 1:
                    # m=1: E_bar_1 = E_hat_1
                    E_bar_by_resolution[res] = E_hat_curr
                    print(f"      m={res}: E_bar_{res} = E_hat_{res} = {E_bar_by_resolution[res]:.6f}")
                else:
                    # m>1: E_bar_m = (alpha*E_bar_{m-1} + (1-alpha)*E_hat_m) / (1-alpha^m)
                    E_bar_prev = E_bar_by_resolution[res - 1]
                    E_bar_curr = (alpha * E_bar_prev + (1 - alpha) * E_hat_curr) / (1 - alpha**res)
                    E_bar_by_resolution[res] = E_bar_curr
                    
                    print(f"      m={res}: E_bar_{res} = (alpha*E_bar_{res-1} + (1-alpha)*E_hat_{res}) / (1-alpha^{res})")
                    print(f"             = ({alpha:.4f}*{E_bar_prev:.6f} + {1-alpha:.4f}*{E_hat_curr:.6f}) / {1-alpha**res:.6f}")
                    print(f"             = {E_bar_curr:.6f}")
            
            # Record history
            for res in E_hat_by_resolution.keys():
                energy_history['E_hat_history'][res].append(E_hat_by_resolution[res])
                energy_history['E_bar_history'][res].append(E_bar_by_resolution[res])
            
            # Step 3: Peak detection: E_bar_{m-1} >= E_hat_m
            if m > 1:
                E_bar_prev = E_bar_by_resolution[m - 1]
                E_hat_curr = E_hat_by_resolution[m]
                
                print(f"\n    Step 3: Peak detection:")
                print(f"      Checking: E_bar_{m-1} >= E_hat_{m} ?")
                print(f"      E_bar_{m-1} = {E_bar_prev:.6f}")
                print(f"      E_hat_{m}   = {E_hat_curr:.6f}")
                
                if E_bar_prev >= E_hat_curr:
                    m_star = m - 1
                    print(f"\n  Peak detected! m* = {m_star}")
                    print(f"      E_bar_{m_star} ({E_bar_prev:.6f}) >= E_hat_{m} ({E_hat_curr:.6f})")
                    break
                else:
                    print(f"      No peak yet: E_bar_{m-1} < E_hat_{m}, continue...")
            
            parent_centers_for_next = current_centers
            m += 1
        
        if m_star is None:
            m_star = max_resolution
            print(f"\n  Max resolution reached, m* = {m_star}")
        
        print("\n" + "="*70)
        print(f"RESULT: m* = {m_star}")
        print("="*70)
        
        # Organize return data
        energy_history['E_hat_by_resolution'] = E_hat_by_resolution
        energy_history['E_bar_by_resolution'] = E_bar_by_resolution
        
        # Build layers_history for visualization
        layers_history = {}
        for m in sorted(energy_history['E_hat_history'].keys()):
            layer_idx = m - 1
            layers_history[layer_idx] = {
                'E_hat': energy_history['E_hat_history'][m],
                'E_bar': energy_history['E_bar_history'][m],
                'resolution': m
            }
        energy_history['layers_history'] = layers_history
        
        return m_star, energy_history
    
    def _compute_ema_alpha(self):
        """Compute EMA coefficient from error threshold"""
        delta_ac = self.config.get('error_threshold', 0.1)
        log_delta = math.log10(max(delta_ac, 1e-10))
        alpha = 2 * math.atan(-log_delta) / math.pi
        alpha = max(0.1, min(0.99, alpha))
        return alpha


def plot_energy_history(energy_history, title_suffix="", save_path=None):
    """Visualize energy history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    layers_history = energy_history['layers_history']
    resolutions = energy_history['resolution']
    
    # Plot 1: E_hat evolution for all layers
    ax1 = axes[0, 0]
    for layer_idx, layer_hist in layers_history.items():
        iterations = list(range(layer_idx + 1, layer_idx + 1 + len(layer_hist['E_hat'])))
        ax1.plot(iterations, layer_hist['E_hat'], 'o-', 
                label=f"Layer {layer_idx} (m={layer_hist['resolution']}) E_hat", 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Energy E_hat (corrected)', fontsize=12)
    ax1.set_title(f'Energy Evolution (All Layers) {title_suffix}', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: E_bar evolution for all layers
    ax2 = axes[0, 1]
    for layer_idx, layer_hist in layers_history.items():
        iterations = list(range(layer_idx + 1, layer_idx + 1 + len(layer_hist['E_bar'])))
        ax2.plot(iterations, layer_hist['E_bar'], 's-', 
                label=f"Layer {layer_idx} E_bar", 
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('EMA E_bar', fontsize=12)
    ax2.set_title(f'EMA Evolution (Peak Detection) {title_suffix}', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Tracking error
    ax3 = axes[1, 0]
    ax3.plot(resolutions, energy_history['max_loss'], 'o-', 
             label='Max |delta|', linewidth=2)
    ax3.plot(resolutions, energy_history['mean_loss'], 's-', 
             label='Mean |delta|', linewidth=2)
    ax3.set_xlabel('Resolution m', fontsize=12)
    ax3.set_ylabel('Tracking Error', fontsize=12)
    ax3.set_title(f'Tracking Error {title_suffix}', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Computation time
    ax4 = axes[1, 1]
    ax4.plot(resolutions, energy_history['odeint_time'], 'o-', 
             linewidth=2, color='purple')
    ax4.set_xlabel('Resolution m', fontsize=12)
    ax4.set_ylabel('odeint Time (s)', fontsize=12)
    ax4.set_title(f'Computation Time {title_suffix}', fontsize=13)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'energy_estimation.pdf'
    
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # 2D system configuration
    def trajectory_2d(t):
        return 0.5 * torch.sin(t) ** 2
    
    def nonlinear_2d(x):
        return (8 * torch.exp(-x[0]**2) * torch.sin(5 * torch.sqrt(x[0]**2 + x[1]**2)) 
                + torch.sin(4 * x[1]))
    
    def manual_deriv_2d(t):
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        cos_2t = torch.cos(2*t)
        return {
            'xr': 0.5 * sin_t * sin_t,
            'xr_dot1': sin_t * cos_t,
            'xr_dot2': cos_2t
        }
    
    config_2d = {
        'system_order': 2,
        'tran_r': 2,
        'dwell_time': 50,
        'lambda_c': 20.0,
        'beta': 20.0,
        'lr':0.0001,
        'sampling_ratio': 0.36,
        'error_threshold': 0.04,
        'max_cache_size': 10000,
        'max_cacheable_j': 5,
        'trajectory_function': trajectory_2d,
        'nonlinear_function': nonlinear_2d,
    }
    
    # 4D system configuration
    config_4d = {
        'system_order': 4,
        'tran_r': 1,
        'dwell_time': 100,
        'lambda_c': 2.0,
        'beta': 10.0,
        'lr':0.0001,
        'sampling_ratio': 0.36,
        'error_threshold': 0.1,
        'max_cache_size': 10000,
        'max_cacheable_j': 5,
        'trajectory_function': None,  # Use default: 0.5*sin(t)^2
        'nonlinear_function': None,   # Use default 4D nonlinear
    }
    
    # Select configuration
    print("="*70)
    print("Select System Configuration:")
    print("  1. 2D system (custom)")
    print("  2. 4D system (from File2)")
    print("="*70)
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    
    if choice == '2':
        config = config_4d
        system_name = "4D"
        save_prefix = "4d"
    else:
        config = config_2d
        system_name = "2D"
        save_prefix = "2d"
    
    initial_state = torch.zeros(config['system_order'] + 1, device=device)
    
    print("\n" + "="*70)
    print(f"Simplified Initial Resolution Estimator - {system_name} System")
    print("="*70)
    print(f"System order: {config['system_order']}")
    print(f"Lambda: {config['lambda_c']}, Beta: {config['beta']}")
    print(f"Dwell time per resolution: {config['dwell_time']}s")
    print("\nKEY CORRECTIONS:")
    print("  1. EMA uses each layer's own update count (not global)")
    print("  2. First update: E_bar = E_hat (ensured)")
    print("  3. Formula: E_bar_k = (alpha*E_bar_{k-1} + (1-alpha)*E_hat_k) / (1-alpha^k)")
    print("="*70)
    
    estimator = SimplifiedResolutionEstimator(config, device)
    
    # Only 2D system needs manual derivatives
    if system_name == "2D":
        estimator.trajectory_processor.set_manual_derivatives(manual_deriv_2d)
    
    start = time.time()
    m_star, hist = estimator.estimate_m_star(
        initial_state, max_resolution=4
    )
    total_time = time.time() - start
    
    print(f"\nTotal time: {total_time:.2f}s")
    
    # Visualization
    plot_energy_history(hist, f"({system_name})", f"energy_estimation_{save_prefix}.pdf")
    
    # Save to Excel
    layers_history = hist['layers_history']
    
    df_main = pd.DataFrame({
        'resolution': hist['resolution'],
        'max_loss': hist['max_loss'],
        'mean_loss': hist['mean_loss'],
        'n_total_bases': hist['n_total_bases'],
        'odeint_time': hist['odeint_time'],
        't_start': hist['t_start'],
        't_end': hist['t_end']
    })
    
    excel_file = f"estimation_results_{save_prefix}.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Main History', index=False)
        
        for layer_idx, layer_hist in layers_history.items():
            iterations = list(range(layer_idx + 1, layer_idx + 1 + len(layer_hist['E_hat'])))
            df_layer = pd.DataFrame({
                'iteration': iterations,
                'E_hat': layer_hist['E_hat'],
                'E_bar': layer_hist['E_bar']
            })
            df_layer.to_excel(writer, sheet_name=f'Layer_{layer_idx}_m{layer_hist["resolution"]}', 
                            index=False)
    
    print(f"\nSaved to {excel_file}")
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Optimal initial resolution: m* = {m_star}")
    print("\nEnergy evolution for each layer:")
    for layer_idx, layer_hist in layers_history.items():
        print(f"\nLayer {layer_idx} (m={layer_hist['resolution']}):")
        for i, (e_hat, e_bar) in enumerate(zip(layer_hist['E_hat'], layer_hist['E_bar'])):
            iteration = layer_idx + 1 + i
            update_num = i + 1
            print(f"  Update #{update_num} (iter {iteration}): E_hat={e_hat:.6f}, E_bar={e_bar:.6f}")
            # Verify first update: E_bar = E_hat
            if update_num == 1 and abs(e_hat - e_bar) > 1e-6:
                print(f"    Warning: First update should have E_bar = E_hat!")
    print("="*70)