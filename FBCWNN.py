"""
FBCWNN (Frequency-Based Constructive Wavelet Neural Network) Implementation
For trajectory tracking in fourth-order dynamic systems
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import Callable, Optional, Dict
from datetime import datetime


class AutomaticTrajectoryProcessor:
    """Trajectory processor - same as AFBCWNN"""
    
    def __init__(self, trajectory_func: Callable, system_order: int, device: str):
        self.trajectory_func = trajectory_func
        self.system_order = system_order
        self.device = device
        self.derivatives_cache = {}
        self.use_manual = False
        self.max_cache_size = 10
        
    def set_manual_derivatives(self, derivatives_func: Callable):
        self.manual_derivatives = derivatives_func
        self.use_manual = True
        
    def compute_derivatives(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
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


class PIDController:
    
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki  
        self.kd = kd
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.dt = 0.01
        
    def compute_control(self, error, error_dot=None):
        """Compute PID control output"""
        proportional = self.kp * error
        
        self.integral_error += error * self.dt
        integral = self.ki * self.integral_error
        
        if error_dot is not None:
            derivative = self.kd * error_dot
        else:
            derivative = self.kd * (error - self.prev_error) / self.dt
            
        self.prev_error = error
        
        return proportional + integral + derivative


class FixedBasisCWNN(nn.Module):
    """Frequency-Based Constructive Wavelet Neural Network - Fourth-Order System"""
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        # System configuration
        self.j = config['j_init']
        self.mu = config['mu']
        self.tran_r = config['tran_r']
        self.max_layers = config['max_layers']
        self.system_order = config.get('system_order', 4)
        self.dimension = self.system_order
        self.lambda_c = config.get('lambda_c', 2.0)
        self.beta = config.get('beta', 10.0)
        
        # Compute control coefficients
        self._compute_control_coefficients()
        
        # Setup trajectory and nonlinear function
        self._setup_trajectory_processor(config)
        self._setup_nonlinear_function(config)
        
        # Phase control
        self.current_phase = 1  # 1: Data Collection, 2: Offline Training, 3: Static Operation
        self.is_trained = False
        self.last_print_time_phase1 = -0.5
        
        # Phase 1: Data collection
        self.training_data = []
        self.trajectory_buffer = []
        self.sample_interval = 0.005
        self.last_sample_time = -self.sample_interval
        self.target_samples = 10000
        
        # Phase 3: Operation data recording (consistent with AFBCWNN format)
        self.training_records = defaultdict(list)
        self.last_print_time = -0.01
        
        # Dwell time mechanism (for accurate max error calculation)
        self.dwell_time = config.get('base_interval', 120)
        self.current_dwell_start = 0.0
        self.dwell_losses = []
        self.dwell_statistics = []
        self.dwell_sample_count = 0
        
        # PID controller
        pid_gains = config.get('pid_gains', {'kp': 10.0, 'ki': 1.0, 'kd': 2.0})
        self.pid_controller = PIDController(
            pid_gains['kp'], 
            pid_gains['ki'], 
            pid_gains['kd']
        )
        
        # Network structure
        self.weights = nn.ParameterList()
        self.centers_list = []
        self.active_indices = []
        self.final_structure = {}
        
        # Precomputation
        self.scaling = 0.5
        self.inv_scaling = 2.0
        self._precompute_powers()
        self._precompute_grid_templates()
        
        print(f"[FBCWNN] Initialized for {self.system_order}D system")
        print(f"Control coefficients: {self.control_coeffs}")
    
    def _compute_control_coefficients(self):
        """Compute control coefficients"""
        d = self.system_order
        lambda_val = self.lambda_c
        
        self.control_coeffs = []
        for k in range(d):
            if k <= d-1:
                n = d - 1
                comb_value = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
            else:
                comb_value = 0
            coeff = comb_value * (lambda_val ** (d - 1 - k))
            self.control_coeffs.append(coeff)
    
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
        """Setup nonlinear function - fourth-order system"""
        nonlinear_func = config.get('nonlinear_function')
        
        if nonlinear_func is None:
            self.compute_nonlinear = self._original_nonlinear_4d
        else:
            self.compute_nonlinear = nonlinear_func
    
    def _original_nonlinear_4d(self, x_est):
        """Nonlinear function for fourth-order system"""
        x0, x1, x2, x3 = x_est[0], x_est[1], x_est[2], x_est[3]
        x_sum = x0 + 2*(x1 + x2 + x3)
        x_partial = x0 + 2*x1 + x2
        x_tail = x1 + 2*x2 + x3
        return -x_sum + (1 - x_partial*x_partial) * x_tail
    
    def _precompute_powers(self):
        """Precompute powers"""
        self.power_cache = {}
        for j in range(10):
            self.power_cache[j] = self.inv_scaling ** j
    
    def _precompute_grid_templates(self):
        """Precompute grid templates"""
        self.grid_cache = {}
        self.mesh_cache = {}
        
        for j in range(1, 4):
            grid_size = self.tran_r * 2**j + 1
            if grid_size <= 50:
                grid = torch.arange(grid_size, device=self.device, dtype=torch.float32)
                self.grid_cache[grid_size] = grid
    
    def _generate_centers(self, j):
        """Generate center points"""
        grid_size = self.tran_r * 2**j + 1
        
        if grid_size in self.mesh_cache:
            return self.mesh_cache[grid_size]
        
        if grid_size in self.grid_cache:
            grid = self.grid_cache[grid_size]
        else:
            grid = torch.arange(grid_size, device=self.device, dtype=torch.float32)
        
        grids = [grid] * self.dimension
        mesh = torch.meshgrid(grids, indexing='ij')
        centers = torch.stack(mesh, dim=-1).reshape(-1, self.dimension).contiguous()
        
        if j <= 2 and len(centers) <= 500:
            self.mesh_cache[grid_size] = centers
        
        return centers
    
    def _wavelet_function_batch(self, x, centers, j, is_base_layer):
        """Batch compute wavelet functions"""
        scale_factor = self.power_cache.get(j, self.inv_scaling ** j)
        
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
    
    # ==================== Phase 1: Data Collection ====================
    
    def collect_training_data(self, t, state):
        """
        Phase 1: Collect trajectory data using PID controller
        
        Note: Only records trajectory (t, x, u), does not directly compute f
        After Phase 1, f is estimated through numerical differentiation
        """
        state = state.detach()
        x_est = state[:self.system_order]
        
        traj = self.trajectory_processor.compute_derivatives(t)
        
        errors = []
        errors.append(x_est[0] - traj['xr'])
        for i in range(1, self.system_order):
            errors.append(x_est[i] - traj[f'xr_dot{i}'])
        
        deta = sum(coeff * error for coeff, error in zip(self.control_coeffs, errors))
        
        # PID control
        error_main = errors[0].item()
        error_dot = errors[1].item() if len(errors) > 1 else 0.0
        pid_output = self.pid_controller.compute_control(error_main, error_dot)
        
        # Control law
        velocity_feedback = sum(self.control_coeffs[i] * errors[i+1] 
                               for i in range(self.system_order - 1))
        velocity_term = traj[f'xr_dot{self.system_order}'] - velocity_feedback
        u_pid = pid_output + velocity_term

        # Sampling control: only record data when sampling interval is met
        current_time = t.item()
        should_sample = (current_time - self.last_sample_time >= self.sample_interval)
        
        # Only record trajectory, do not compute f
        if should_sample:
            self.trajectory_buffer.append({
                't': t.item(),
                'x': x_est.clone().detach(),
                'u': u_pid.clone().detach(),
                'tracking_error': abs(deta.item())
            })
            self.last_sample_time = current_time
        
        # True system response (for simulation)
        f_true = self.compute_nonlinear(x_est)
        
        # State derivatives
        X = torch.zeros_like(state)
        for i in range(self.system_order - 1):
            X[i] = x_est[i + 1]
        X[self.system_order - 1] = f_true + u_pid
        
        return X.detach()
    
    def process_trajectory_data(self):
        """
        Post-Phase 1 processing: Estimate f through numerical differentiation
        
        This is the key step:
        1. Perform numerical differentiation on recorded trajectory to get ẋ_d
        2. Use f(x) ≈ ẋ_d - u to estimate nonlinear term
        3. Construct training data (x, f_estimated)
        """
        print(f"\n[Data Processing] Processing {len(self.trajectory_buffer)} trajectory points...")
        
        if len(self.trajectory_buffer) < 3:
            print("Error: Too few trajectory points for numerical differentiation!")
            return False
        
        # Extract data
        times = torch.tensor([d['t'] for d in self.trajectory_buffer])
        states = torch.stack([d['x'] for d in self.trajectory_buffer])
        controls = torch.stack([d['u'] for d in self.trajectory_buffer])
        
        # Numerical differentiation: compute ẋ_d (system highest-order derivative)
        x_d = states[:, self.system_order - 1]
        
        # Use central difference (more accurate)
        x_dot_d = torch.zeros_like(x_d)
        dt = times[1] - times[0]
        
        # Boundary points use forward/backward difference
        x_dot_d[0] = (x_d[1] - x_d[0]) / dt
        x_dot_d[-1] = (x_d[-1] - x_d[-2]) / dt
        
        # Interior points use central difference
        for i in range(1, len(x_d) - 1):
            x_dot_d[i] = (x_d[i+1] - x_d[i-1]) / (2 * dt)
        
        # Optional: add numerical differentiation noise
        if self.config.get('add_differentiation_noise', False):
            noise_level = self.config.get('diff_noise_level', 0.01)
            diff_noise = torch.randn_like(x_dot_d) * noise_level
            x_dot_d = x_dot_d + diff_noise
            print(f"  Added differentiation noise: level={noise_level}")
        
        # Estimate nonlinear term: f ≈ ẋ_d - u
        for i in range(len(self.trajectory_buffer)):
            x = states[i]
            u = controls[i]
            x_dot_d_i = x_dot_d[i]
            
            # Key: derivative from numerical differentiation minus control input
            f_estimated = x_dot_d_i - u
            
            self.training_data.append({
                'x': x,
                'f_x': f_estimated,
                't': times[i].item(),
                'tracking_error': self.trajectory_buffer[i]['tracking_error']
            })
        
        print(f"  Numerical differentiation completed")
        print(f"  Generated {len(self.training_data)} training samples")
        print(f"  f(x) estimated via: f ≈ dx_{self.system_order-1}/dt - u")
        
        return True
    
    # ==================== Phase 2: Offline Training ====================
    
    def start_offline_training(self):
        """Phase 2: Offline training"""
        print(f"\n{'='*70}")
        print("Phase 2: Offline Training (Progressive Structure Expansion)")
        print(f"{'='*70}")
        
        # First process trajectory data (numerical differentiation)
        success = self.process_trajectory_data()
        if not success:
            return False
        
        print(f"Training data prepared: {len(self.training_data)} samples")
        
        if len(self.training_data) < 10:
            print("Error: Too few training samples!")
            return False
        
        # Extract training data
        X_data = torch.stack([data['x'] for data in self.training_data])
        F_data = torch.stack([data['f_x'] for data in self.training_data])
        
        # Progressive expansion from m=1
        self._progressive_structure_expansion(X_data, F_data)
        
        self.is_trained = True
        self.current_phase = 3
        
        print(f"\n{'='*70}")
        print("Phase 2 Completed!")
        print(f"  Final structure: {self.final_structure.get('n_layers', 0)} layers")
        print(f"  Total parameters: {self.final_structure.get('total_params', 0)}")
        print(f"{'='*70}\n")
        
        return True
    
    def _progressive_structure_expansion(self, X_data, F_data):
        """Progressive network structure expansion from m=1 (consistent with AFBCWNN expansion)"""
        target_error = self.config.get('offline_target_error', 0.1)
        max_resolution = self.config.get('max_resolution', 3)
        
        current_m = 1
        
        # Initialize first layer: V1 and W1
        print(f"\n[Expansion] Starting from resolution m={current_m}")
        centers_1 = self._generate_centers(current_m)
        
        # Limit initial centers for fourth-order system
        if len(centers_1) > 100:
            indices = torch.randperm(len(centers_1))[:100]
            centers_1 = centers_1[indices]
        
        self.centers_list.append(centers_1)
        self.active_indices.append(torch.arange(len(centers_1), device=self.device))
        
        # First layer contains phi + psi
        weight_size = 2 * len(centers_1)
        self.weights.append(nn.Parameter(torch.zeros(weight_size, device=self.device)))
        
        print(f"  Layer 0 (m={current_m}): {len(centers_1)} centers, {weight_size} weights (phi+psi)")
        
        # Train current structure
        current_error = self._train_current_structure(X_data, F_data)
        print(f"  Initial error: {current_error:.6f}")
        
        # Progressive expansion (using parent-child mechanism)
        layer_idx = 1
        while current_error > target_error and current_m < max_resolution:
            current_m += 1
            parent_layer = layer_idx - 1
            
            print(f"\n[Expansion] Adding layer {layer_idx} at resolution m={current_m}")
            
            # Select high-energy bases from parent layer
            parent_selected_indices = self._energy_separate_offline(
                parent_layer, X_data, F_data
            )
            
            if len(parent_selected_indices) == 0:
                print("  No parent bases selected, stopping expansion")
                break
            
            # Generate new resolution centers
            new_centers = self._generate_centers(current_m)
            
            # Generate child centers from parent centers (consistent with AFBCWNN)
            parent_centers = self.centers_list[parent_layer][parent_selected_indices]
            selected_child_indices = self._generate_child_centers(
                parent_centers, new_centers, current_m
            )
            
            if len(selected_child_indices) == 0:
                print("  No child centers generated, stopping expansion")
                break
            
            # Add new layer
            self.centers_list.append(new_centers)
            self.active_indices.append(selected_child_indices)
            
            # New layer has only psi (no phi)
            new_weight_size = len(selected_child_indices)
            self.weights.append(nn.Parameter(torch.zeros(new_weight_size, device=self.device)))
            
            # Retrain
            current_error = self._train_current_structure(X_data, F_data)
            print(f"  Current error: {current_error:.6f}")
            
            layer_idx += 1
            
            if layer_idx >= self.max_layers:
                print(f"  Reached max layers ({self.max_layers}), stopping")
                break
        
        # Fix weights
        for w in self.weights:
            w.requires_grad = False
        
        self.final_structure = {
            'n_layers': len(self.weights),
            'total_params': sum(w.numel() for w in self.weights),
            'final_error': current_error,
            'resolutions': list(range(1, current_m + 1))
        }
    
    def _energy_separate_offline(self, parent_layer, X_data, F_data):
        """
        Energy separation: select high-energy bases from parent layer
        """
        if parent_layer == 0:
            weights = self.weights[parent_layer].data
            n_centers = len(self.centers_list[parent_layer])
            psi_weights = weights[n_centers:]
        else:
            weights = self.weights[parent_layer].data
            psi_weights = weights
        
        # Compute energy
        energy = psi_weights * psi_weights
        total_energy = energy.sum()
        
        if total_energy < 1e-10:
            return torch.tensor([], device=self.device, dtype=torch.long)
        
        # Select top mu proportion of high-energy bases
        n_elements = len(energy)
        k = max(1, int(n_elements * self.mu))
        
        _, top_indices = torch.topk(energy, min(k, n_elements))
        
        return top_indices
    
    def _generate_child_centers(self, parent_centers, new_centers, new_j):
        """
        Generate child centers from parent centers (consistent with AFBCWNN)
        
        For each parent center, generate 2^d child centers (doubling + offset)
        """
        if len(parent_centers) == 0:
            return torch.tensor([], device=self.device, dtype=torch.long)
        
        max_pos = self.tran_r * (self.inv_scaling ** new_j)
        parent_coords = parent_centers.round()
        
        # Create offset template: 2^d combinations (d-dimensional Cartesian product of [-1,0])
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
        
        # Generate child centers for each parent center
        for p_coord in parent_coords:
            # Double + offset: p_coord * 2 + offset
            candidates = (p_coord.unsqueeze(0) * 2 + offsets).clamp(0.0, max_pos_value)
            
            # Find these candidate centers in new resolution grid
            for candidate in candidates:
                diffs = torch.abs(new_centers - candidate).sum(dim=1)
                matches = torch.nonzero(diffs < 1e-5).view(-1)
                if len(matches) > 0:
                    selected_indices.append(matches[0].item())
        
        # Remove duplicates
        if selected_indices:
            return torch.unique(torch.tensor(selected_indices, device=self.device, dtype=torch.long))
        else:
            return torch.tensor([], device=self.device, dtype=torch.long)
    
    def _train_current_structure(self, X_data, F_data):
        """Train current network structure"""
        n_samples = len(X_data)
        batch_size = min(32, n_samples)
        epochs = 50
        lr = 0.0001
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_data[batch_indices]
                batch_F = F_data[batch_indices]
                
                optimizer.zero_grad()
                predictions = torch.stack([self._forward_inference(x) for x in batch_X])
                loss = criterion(predictions, batch_F)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
        
        # Return final error
        with torch.no_grad():
            predictions = torch.stack([self._forward_inference(x) for x in X_data])
            final_error = torch.mean((predictions - F_data) ** 2).sqrt().item()
        
        return final_error
    
    def _forward_inference(self, x):
        """Forward inference"""
        if x.dim() == 0:
            return torch.tensor(0.0, device=self.device)
        
        wavelet_sum = torch.tensor(0.0, device=self.device)
        
        if len(self.weights) == 0:
            return wavelet_sum
        
        for layer_idx in range(len(self.weights)):
            centers = self.centers_list[layer_idx]
            weights = self.weights[layer_idx]
            
            if layer_idx == 0:
                j = 1
                phi, psi = self._wavelet_function_batch(x, centers, j, True)
                n_centers = len(centers)
                phi_weights = weights[:n_centers]
                psi_weights = weights[n_centers:]
                wavelet_sum += torch.dot(phi_weights, phi)
                wavelet_sum += torch.dot(psi_weights, psi)
            else:
                j = layer_idx + 1
                active_idx = self.active_indices[layer_idx]
                if len(active_idx) > 0:
                    active_centers = centers[active_idx]
                    _, psi = self._wavelet_function_batch(x, active_centers, j, False)
                    if len(psi) == len(weights):
                        wavelet_sum += torch.dot(weights, psi)
        
        return wavelet_sum
    
    # ==================== Phase 3: Static Operation (in dynamic system) ====================
    
    def static_operation(self, t, state):
        """Phase 3: Run with FBCWNN in dynamic system"""
        state = state.detach()
        x_est = state[:self.system_order]
        
        traj = self.trajectory_processor.compute_derivatives(t)
        
        # Compute tracking error
        errors = []
        errors.append(x_est[0] - traj['xr'])
        for i in range(1, self.system_order):
            errors.append(x_est[i] - traj[f'xr_dot{i}'])
        
        deta = sum(coeff * error for coeff, error in zip(self.control_coeffs, errors))
        
        # Use fixed CWNN approximation
        f_hat = self._forward_inference(x_est)
        
        # Control law
        velocity_feedback = sum(self.control_coeffs[i] * errors[i+1] 
                               for i in range(self.system_order - 1))
        velocity_term = traj[f'xr_dot{self.system_order}'] - velocity_feedback
        control_input = -self.beta * deta - f_hat + velocity_term
        
        # State derivatives
        X = torch.zeros_like(state)
        for i in range(self.system_order - 1):
            X[i] = x_est[i + 1]
        X[self.system_order - 1] = self.compute_nonlinear(x_est) + control_input
        
        # Record data (this is the actual training record for final analysis)
        self._record_phase3_data(t, deta)
        
        return X.detach()
    
    def _record_phase3_data(self, t, deta):
        """Record Phase 3 operation data (consistent with AFBCWNN format)"""
        current_time = t.item()
        deta_abs = abs(deta.item())
        
        # Collect all errors within dwell period
        self.dwell_losses.append(deta_abs)
        self.dwell_sample_count += 1
        
        # Check if a dwell time is completed
        self._check_dwell_time(t)
        
        # Record every 0.05s (for plotting)
        params_count = sum(w.numel() for w in self.weights)
        
        if current_time - self.last_print_time >= 0.05:
            self.training_records['time'].append(current_time)
            self.training_records['loss'].append(deta.item())
            self.training_records['params_count'].append(params_count)
            self.training_records['stage'].append(3)
            
            print(f"[t={current_time:.2f}] Phase 3 (Static), loss={deta_abs:.6f}, "
                  f"params={params_count} (fixed)")
            self.last_print_time = current_time
    
    def _check_dwell_time(self, t):
        """Check if a dwell time is completed (consistent with AFBCWNN)"""
        current_time = t.item()
        
        if current_time >= self.current_dwell_start + self.dwell_time - 1e-6:
            if self.dwell_losses:
                max_error = max(self.dwell_losses)
                avg_error = np.mean(self.dwell_losses)
                min_error = min(self.dwell_losses)
                
                # Save complete statistics for this dwell period
                self.dwell_statistics.append({
                    'start_time': self.current_dwell_start,
                    'end_time': current_time,
                    'dwell_time': self.dwell_time,
                    'max_error': max_error,
                    'avg_error': avg_error,
                    'min_error': min_error,
                    'sample_count': len(self.dwell_losses),
                    'phase': 3
                })
                
                print(f"\n[Dwell Time] Window completed at t={current_time:.2f}")
                print(f"  ΔT={self.dwell_time:.2f}, max_error={max_error:.6f}, avg_error={avg_error:.6f}")
                print(f"  Samples in this dwell: {self.dwell_sample_count}\n")
                
                self.dwell_sample_count = 0
            
            # Start new dwell period
            self.current_dwell_start = current_time
            self.dwell_losses.clear()
    
    # ==================== Main forward function ====================
    
    def forward(self, t, state):
        """Forward propagation for dynamic system (call different functions based on phase)"""
        if self.current_phase == 1:
            return self.collect_training_data(t, state)
        elif self.current_phase == 3:
            return self.static_operation(t, state)
        else:
            return torch.zeros_like(state)
    
    # ==================== Save and visualization ====================
    
    def save_to_excel(self, filename=None):
        """Save training data to Excel"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mu_str = f"{self.mu:.2f}".replace('.', 'p')
            filename = f"FBCWNN_{self.system_order}D_mu{mu_str}_dt{self.dwell_time}_{timestamp}.xlsx"
        
        df = pd.DataFrame(dict(self.training_records))
        df.sort_values(by='time', inplace=True)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Training data (recorded every 0.05s)
            df.to_excel(writer, sheet_name='Training Data', index=False)
            
            # Sheet 2: Configuration parameters
            config_df = pd.DataFrame({
                'Parameter': ['system_order', 'lambda_c', 'beta', 'j_init', 'mu',
                            'method', 'n_layers', 'total_params', 'dwell_time'],
                'Value': [self.system_order, self.lambda_c, self.beta,
                        self.j, self.mu, 'FBCWNN',
                        self.final_structure.get('n_layers', 0),
                        self.final_structure.get('total_params', 0),
                        self.dwell_time]
            })
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
            
            # Sheet 3: Phase statistics
            phase_stats = []
            if self.training_data:
                phase_stats.append({
                    'Phase': 'Phase 1 (Data Collection)',
                    'Duration': self.training_data[-1]['t'],
                    'Samples': len(self.training_data),
                    'Mean Error': np.mean([d['tracking_error'] for d in self.training_data]),
                    'Max Error': np.max([d['tracking_error'] for d in self.training_data])
                })
            
            phase_stats.append({
                'Phase': 'Phase 2 (Offline Training)',
                'Final Error': self.final_structure.get('final_error', 0),
                'Layers': self.final_structure.get('n_layers', 0),
                'Total Params': self.final_structure.get('total_params', 0)
            })
            
            if self.training_records['time']:
                phase3_errors = [abs(e) for e in self.training_records['loss']]
                phase_stats.append({
                    'Phase': 'Phase 3 (Static Operation)',
                    'Duration': self.training_records['time'][-1],
                    'Mean Error': np.mean(phase3_errors),
                    'Max Error': np.max(phase3_errors),
                    'Min Error': np.min(phase3_errors)
                })
            
            phase_df = pd.DataFrame(phase_stats)
            phase_df.to_excel(writer, sheet_name='Phase Statistics', index=False)
            
            # Sheet 4: Dwell window statistics (consistent with AFBCWNN's Window Statistics)
            if self.dwell_statistics:
                dwell_stats = []
                for i, stat in enumerate(self.dwell_statistics):
                    dwell_stats.append({
                        'Window Number': i + 1,
                        'Start Time (s)': stat['start_time'],
                        'End Time (s)': stat['end_time'],
                        'Dwell Time': stat['dwell_time'],
                        'Mean |δ|': stat['avg_error'],
                        'Max |δ|': stat['max_error'],
                        'Min |δ|': stat['min_error'],
                        'Phase': stat['phase'],
                        'Data Points (full)': stat['sample_count']
                    })
                
                dwell_df = pd.DataFrame(dwell_stats)
                dwell_df.to_excel(writer, sheet_name='Dwell Statistics', index=False)
                print(f"  - Dwell statistics saved ({len(dwell_stats)} windows)")
                
                # Print key statistics
                max_errors = [w['Max |δ|'] for w in dwell_stats]
                print(f"  - Max errors across windows: min={min(max_errors):.6f}, max={max(max_errors):.6f}")
        
        print(f"\nTraining data saved to {filename}")
        return filename
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.training_records['time']:
            print("No Phase 3 data to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        data = dict(self.training_records)
        
        axes[0, 0].plot(data['time'], [abs(e) for e in data['loss']])
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('|Loss|')
        axes[0, 0].set_title('Tracking Error (FBCWNN - Phase 3)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].plot(data['time'], data['params_count'])
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Number of Parameters')
        axes[0, 1].set_title('Fixed Parameters')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist([abs(e) for e in data['loss']], bins=50, alpha=0.7)
        axes[1, 0].set_xlabel('|Loss|')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].text(0.5, 0.5, 
                       f"Method: FBCWNN\n"
                       f"System Order: {self.system_order}\n"
                       f"Total Params: {self.final_structure.get('total_params', 0)}\n"
                       f"Layers: {self.final_structure.get('n_layers', 0)}\n"
                       f"Mean Error: {np.mean([abs(e) for e in data['loss']]):.6f}",
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        plot_dir = "loss_plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mu_str = f"{self.mu:.2f}".replace('.', 'p')
        plot_path = os.path.join(plot_dir, 
                                f"FBCWNN_{self.system_order}D_mu{mu_str}_{timestamp}.png")
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training progress plot saved to {plot_path}")
        return plot_path


# ==================== Main experiment function ====================

def run_fbcwnn_experiment():
    """Run complete three-phase FBCWNN experiment (fourth-order system)"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Configuration (consistent with AFBCWNN fourth-order system)
    config = {
        'j_init': 1,
        'mu': 1/3,
        'tran_r': 1,
        'max_layers': 5,
        
        # System parameters
        'system_order': 4,
        'lambda_c': 2.0,
        'beta': 10.0,
        'sample_interval': 0.005,
        
        # Trajectory and nonlinear function (default)
        'trajectory_function': None,
        'nonlinear_function': None,
        
        # PID parameters
        'pid_gains': {'kp': 10.0, 'ki': 1.0, 'kd': 2.0},
        
        # Phase 2 parameters
        'offline_target_error': 0.1,
        'max_resolution': 2,
        
        # Dwell time parameters (consistent with AFBCWNN)
        'base_interval': 120,
        
        # Optional: Noise parameters (simulate real measurements)
        # 'add_differentiation_noise': True,
        # 'diff_noise_level': 0.01
    }
    
    print("\n" + "="*70)
    print("FBCWNN Configuration")
    print("="*70)
    print(f"System Order: {config['system_order']}")
    print(f"Data Collection Method: PID + Numerical Differentiation")
    print(f"  Phase 1: Record trajectory (t, x, u)")
    print(f"  Data Processing: Numerical differentiation")
    print(f"    - Compute: ẋ_d ≈ diff(x_d) / dt")
    print(f"    - Estimate: f(x) ≈ ẋ_d - u")
    print(f"  Differentiation noise: {'Enabled' if config.get('add_differentiation_noise', False) else 'Disabled'}")
    if config.get('add_differentiation_noise', False):
        print(f"  Noise level: {config.get('diff_noise_level', 0.01)}")
    print("="*70 + "\n")
    
    model = FixedBasisCWNN(config, device)
    
    # ========== Phase 1: Data Collection ==========
    print("="*70)
    print("Phase 1: Data Collection with PID Controller")
    print("="*70)
    
    initial_state = torch.zeros(config['system_order'] + 1, device=device)
    data_collection_time = 10
    n_steps = int(data_collection_time * 3)
    
    t_span_phase1 = torch.linspace(0, data_collection_time, n_steps).to(device)
    
    print(f"Running PID controller for {data_collection_time}s...")
    
    states_phase1 = odeint(
        model,
        initial_state,
        t_span_phase1,
        method='dopri5',
        rtol=1e-6,
        atol=1e-7
    )
    
    print(f"\nPhase 1 Completed: {len(model.trajectory_buffer)} trajectory points collected")
    
    # ========== Phase 2: Offline Training ==========
    model.current_phase = 2
    success = model.start_offline_training()
    
    if not success:
        print("Phase 2 failed!")
        return None
    
    # ========== Phase 3: Static Operation ==========
    print("="*70)
    print("Phase 3: Static Operation with Fixed CWNN")
    print("="*70)
    
    initial_state = torch.zeros(config['system_order'] + 1, device=device)
    static_time = 961
    n_steps_phase3 = int(static_time * 3)
    
    t_span_phase3 = torch.linspace(0, static_time, n_steps_phase3).to(device)
    
    print(f"Running dynamic system with fixed CWNN for {static_time}s...\n")
    
    states_phase3 = odeint(
        model,
        initial_state,
        t_span_phase3,
        method='dopri5',
        rtol=1e-6,
        atol=1e-7
    )
    
    # ========== Save Results ==========
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    
    excel_file = model.save_to_excel()
    plot_file = model.plot_training_progress()
    
    # Statistics (using dwell_statistics for accurate max error)
    print(f"\nPhase 3 Final Statistics:")
    
    if model.dwell_statistics:
        # Use dwell statistics (accurate max errors)
        all_max_errors = [stat['max_error'] for stat in model.dwell_statistics]
        all_avg_errors = [stat['avg_error'] for stat in model.dwell_statistics]
        all_min_errors = [stat['min_error'] for stat in model.dwell_statistics]
        
        print(f"  Total dwell windows: {len(model.dwell_statistics)}")
        print(f"  Dwell time: {model.dwell_time}s")
        print(f"  Overall max error (across all windows): {max(all_max_errors):.6f}")
        print(f"  Overall avg error (across all windows): {np.mean(all_avg_errors):.6f}")
        print(f"  Overall min error (across all windows): {min(all_min_errors):.6f}")
        
        print(f"\n  Per-window max errors:")
        for i, stat in enumerate(model.dwell_statistics):
            print(f"    Window {i+1} [{stat['start_time']:.0f}s-{stat['end_time']:.0f}s]: "
                  f"max={stat['max_error']:.6f}, avg={stat['avg_error']:.6f}")
    else:
        # Fallback: use training_records
        if model.training_records['time']:
            phase3_errors = [abs(e) for e in model.training_records['loss']]
            print(f"  Duration: {model.training_records['time'][-1]:.2f}s")
            print(f"  Mean error: {np.mean(phase3_errors):.6f}")
            print(f"  Max error (sampled): {np.max(phase3_errors):.6f}")
            print(f"  Min error (sampled): {np.min(phase3_errors):.6f}")
            print(f"  Note: These are sampled errors (every 0.05s), not true max errors")
    
    print("\n" + "="*70)
    print("FBCWNN Experiment Completed!")
    print("="*70)
    print(f"\nFiles saved:")
    print(f"  Excel: {excel_file}")
    print(f"  Plot: {plot_file}")
    print("\nReady for comparison with AFBCWNN results.")
    print("="*70 + "\n")
    
    return model


if __name__ == "__main__":
    model = run_fbcwnn_experiment()