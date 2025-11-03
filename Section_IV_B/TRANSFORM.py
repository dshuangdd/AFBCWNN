"""
Transformer-based Adaptive Controller for 4th-Order Nonlinear System
Fixed-architecture Transformer with LSTM temporal encoding and online learning
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.integrate import solve_ivp
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from datetime import datetime


# ======================================================
# Configuration
# ======================================================
class SystemConfig:
    """System configuration and parameters"""
    def __init__(self, dt=0.01, dwell_time=120.0, max_time=720.0):
        self.dt = dt
        self.dwell_time = dwell_time
        self.max_time = max_time
        self.dwell_steps = int(round(dwell_time / dt))
        
        # System parameters
        self.system_order = 4
        self.lambda_c = 2.0
        
        # Learning parameters
        self.fast_update_interval = 50
        self.pretrain_duration = 30.0
        self.pretrain_steps = int(self.pretrain_duration / dt)
        
        # Control parameters
        self.control_bounds = (-10.0, 10.0)
        self.safety_bounds = (-8.0, 8.0)
        self.target_error = 0.1
        
        # ODE solver parameters
        self.ode_method = 'RK45'
        self.rtol = 1e-6
        self.atol = 1e-7
        
        # Data recording interval
        self.record_interval = 0.25
        self.record_steps = int(round(self.record_interval / dt))
        
        # PID parameters design
        self._design_pid_parameters()
    
    def _design_pid_parameters(self):
        """
        Design PID parameters based on augmented error dynamics
        
        Augmented error δ = (d/dt + λ)^(d-1) * (x1 - xr) creates second-order-like
        dynamics for control design, though the full system remains 4th-order.
        
        With λ=2.0, the PID gains are adjusted to account for this pre-damping effect.
        """
        # Desired performance specifications
        settling_time = 4.0
        zeta = 0.7
        
        # Effective natural frequency
        omega_n = 4.0 / (zeta * settling_time)
        
        # Base PID parameter calculation
        kp_base = omega_n ** 2
        kd_base = 2 * zeta * omega_n
        ki_base = 0.1 * kp_base
        
        # Lambda compensation factor
        lambda_factor = 1.0 + 0.3 * self.lambda_c
        
        # Final parameters (rounded to practical values)
        self.kp = 3.5
        self.kd = 3.0
        self.ki = 0.3
        
        print(f"\n{'='*70}")
        print("PID Parameter Design (Safety Fallback Only)")
        print(f"{'='*70}")
        print(f"System: 4th-order, λ={self.lambda_c}")
        print(f"Design: ζ={zeta}, ωn={omega_n:.2f} rad/s, Ts={settling_time}s")
        print(f"Parameters: Kp={self.kp}, Kd={self.kd}, Ki={self.ki}")
        print(f"Note: PID used only for data collection and safety fallback")
        print(f"{'='*70}\n")


# ======================================================
# Fourth-Order Nonlinear System
# ======================================================
class FourthOrderSystemODE:
    """4th-order nonlinear system dynamics"""
    
    def __init__(self, config):
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset system to initial state"""
        self.state = np.zeros(4, dtype=np.float64)
        self.time = 0.0
        return self.state
    
    def get_reference_state(self, t):
        """
        Reference trajectory and its derivatives
        yr(t) = 0.5 * sin²(t)
        """
        xr = 0.5 * np.sin(t) ** 2
        xr_dot = np.sin(t) * np.cos(t)
        xr_ddot = np.cos(2*t)
        xr_dddot = -2 * np.sin(2*t)
        
        return np.array([xr, xr_dot, xr_ddot, xr_dddot], dtype=np.float64)
    
    def nonlinear_function(self, x):
        """
        System nonlinearity f(x)
        f(x) = -(x1 + 2x2 + 2x3 + 2x4) + [1 - (x1 + 2x2 + x3)²](x2 + 2x3 + x4)
        """
        x1, x2, x3, x4 = x
        
        term1 = -(x1 + 2*x2 + 2*x3 + 2*x4)
        term2 = (1 - (x1 + 2*x2 + x3)**2) * (x2 + 2*x3 + x4)
        
        return term1 + term2
    
    def dynamics(self, t, x, u):
        """System dynamics: x' = [x2, x3, x4, f(x) + u]"""
        dx = np.zeros(4, dtype=np.float64)
        dx[0] = x[1]
        dx[1] = x[2]
        dx[2] = x[3]
        dx[3] = self.nonlinear_function(x) + u
        
        return dx
    
    def step(self, u):
        """Integrate one time step using ODE solver"""
        def dynamics_with_control(t, x):
            return self.dynamics(t, x, u)
        
        sol = solve_ivp(
            dynamics_with_control,
            [self.time, self.time + self.config.dt],
            self.state,
            method=self.config.ode_method,
            rtol=self.config.rtol,
            atol=self.config.atol,
            dense_output=True
        )
        
        self.state = sol.y[:, -1].astype(np.float64)
        self.time += self.config.dt
        ref = self.get_reference_state(self.time)
        
        return self.state, ref


# ======================================================
# Fixed Transformer Architecture
# ======================================================
class FixedTransformer(nn.Module):
    """Fixed-architecture Transformer for 4D system control"""
    
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Temporal encoder (bidirectional LSTM)
        self.temporal_encoder = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # Fixed Transformer layers
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim * 2,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """Forward pass through Transformer"""
        # Input encoding
        h = self.input_encoder(x)
        
        # Temporal encoding
        h, _ = self.temporal_encoder(h)
        
        # Transformer layers
        for block in self.transformer_blocks:
            h = block(h)
        
        # Output
        out = self.output_head(h[:, -1, :])
        return torch.tanh(out) * 5.0
    
    def get_info(self):
        """Get network information"""
        param_count = sum(p.numel() for p in self.parameters())
        return {
            'layers': self.num_layers,
            'parameters': param_count,
            'architecture': f"Fixed {self.num_layers}-layer Transformer (4D)"
        }


# ======================================================
# Transformer Controller
# ======================================================
class TransformerController:
    """Transformer controller with PID safety fallback"""
    def __init__(self, config):
        self.config = config
        
        # Fixed-architecture Transformer
        self.model = FixedTransformer(input_dim=10, hidden_dim=64, num_layers=3)
        
        # Optimizers (different for pretrain vs online)
        self.pretrain_optimizer = optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        self.online_optimizer = optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.online_optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
        )
        
        # Operating mode
        self.mode = 'collect'
        
        # Data management (sequence buffer for Transformer)
        self.sequence_buffer = deque(maxlen=30)
        self.training_data = deque(maxlen=5000)
        
        # Statistics
        self.step_count = 0
        self.window_count = 0
        self.dwell_counter = 0
        
        self.stats = {
            'transformer_actions': 0,
            'safety_interventions': 0,
            'fallback_activations': 0,
            'window_errors': [],
            'window_max_errors': [],
            'learning_rates': [],
            'mode_history': []
        }
        
        self.results = {
            'time': [],
            'delta': [],
            'abs_delta': [],
            'control': [],
            'mode': []
        }
        
        # PID state
        self.integral_error = 0.0
        self.prev_error = 0.0
        
        # Previous control input
        self.u_prev = 0.0
        
        # Print network info
        net_info = self.model.get_info()
        print(f"\n[Transformer Network]")
        print(f"  Architecture: {net_info['architecture']}")
        print(f"  Parameters: {net_info['parameters']:,}")
        print(f"  Features: LSTM + {net_info['layers']} Transformer layers")
    
    def compute_augmented_error(self, state, ref):
        """
        Compute augmented tracking error
        
        For 4th-order system (d=4):
        δ = (d/dt + λ)³ * (x1 - xr)
        
        Expanded using binomial theorem:
        δ = λ³(x1-xr) + 3λ²(x2-xr') + 3λ(x3-xr'') + (x4-xr''')
        """
        lam = self.config.lambda_c
        
        e1 = state[0] - ref[0]
        e2 = state[1] - ref[1]
        e3 = state[2] - ref[2]
        e4 = state[3] - ref[3]
        
        delta = lam**3 * e1 + 3*lam**2 * e2 + 3*lam * e3 + e4
        
        return delta
    
    def construct_observation(self, state, ref):
        """Construct 10-dimensional network input"""
        delta = self.compute_augmented_error(state, ref)
        
        obs = np.array([
            state[0], state[1], state[2], state[3],
            ref[0], ref[1], ref[2], ref[3],
            self.u_prev,
            delta
        ], dtype=np.float32)
        
        return obs
    
    def pid_control(self, delta):
        """
        PID safety controller (for data collection and fallback only)
        
        Direct PD control on augmented error δ with lambda pre-damping accounted for
        """
        delta_dot = (delta - self.prev_error) / self.config.dt
        
        u = -self.config.kp * delta - self.config.kd * delta_dot
        
        self.prev_error = delta
        return np.clip(u, *self.config.control_bounds)
    
    def select_action(self, obs, deterministic=False):
        """Select action from Transformer (with sequence processing)"""
        # Update sequence buffer
        self.sequence_buffer.append(obs)
        
        if len(self.sequence_buffer) < 20:
            return 0.0
        
        # Prepare sequence input
        seq = np.array(list(self.sequence_buffer)[-20:])
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
        
        if self.mode == 'collect' or self.mode == 'pretrain':
            # Random exploration during pretraining
            action = np.random.uniform(-3.0, 3.0)
        else:
            with torch.no_grad():
                action = self.model(seq_tensor).cpu().numpy()[0, 0]
        
        return action
    
    def update_model(self):
        """Update Transformer model"""
        if len(self.training_data) < 100:
            return 0.0
        
        # Sample batch
        batch_size = min(64, len(self.training_data))
        indices = np.random.choice(len(self.training_data), batch_size, replace=False)
        batch = [self.training_data[i] for i in indices]
        
        sequences = torch.FloatTensor(np.array([item[0] for item in batch]))
        targets = torch.FloatTensor(np.array([[item[1]] for item in batch]))
        
        # Forward pass
        predictions = self.model(sequences)
        loss = F.mse_loss(predictions, targets)
        
        # Backward pass
        if self.mode == 'pretrain':
            self.pretrain_optimizer.zero_grad()
        else:
            self.online_optimizer.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        if self.mode == 'pretrain':
            self.pretrain_optimizer.step()
        else:
            self.online_optimizer.step()
        
        return loss.item()
    
    def pretrain(self):
        """Pretrain Transformer with collected data"""
        print(f"\n{'='*70}")
        print(f"Pretraining Transformer (Data: {len(self.training_data)} samples)")
        print(f"{'='*70}")
        
        for epoch in range(100):
            if len(self.training_data) < 100:
                continue
            
            loss = self.update_model()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/100, Loss: {loss:.6f}")
        
        print("Pretraining completed!\n")
        self.mode = 'transformer'


# ======================================================
# Main Evaluation Function
# ======================================================
def evaluate_controller(config):
    """Main Transformer controller evaluation"""
    env = FourthOrderSystemODE(config)
    controller = TransformerController(config)
    
    state = env.reset()
    ref = env.get_reference_state(0.0)
    obs = controller.construct_observation(state, ref)
    
    # Window statistics
    dwell_errors = []
    window_start_time = 0.0
    
    print(f"\n{'='*70}")
    print("Transformer Controller Evaluation - 4D Nonlinear System")
    print(f"{'='*70}\n")
    
    for step in range(int(config.max_time / config.dt)):
        t = step * config.dt
        
        # Compute augmented error
        delta = controller.compute_augmented_error(state, ref)
        abs_delta = abs(delta)
        
        # Collect dwell window errors
        dwell_errors.append(abs_delta)
        
        # Select control action
        if controller.mode == 'collect':
            u = controller.pid_control(delta)
            mode = 'PID'
            controller.sequence_buffer.append(obs)
        elif controller.mode == 'pretrain':
            u = controller.select_action(obs, deterministic=False)
            u = np.clip(u, *config.control_bounds)
            mode = 'Explore'
        else:
            u = controller.select_action(obs, deterministic=True)
            
            # Safety check
            if abs(delta) > 1.0 or np.isnan(u) or np.isinf(u):
                u = controller.pid_control(delta)
                mode = 'Fallback'
                controller.stats['fallback_activations'] += 1
            else:
                u = np.clip(u, *config.safety_bounds)
                mode = 'Transformer'
                controller.stats['transformer_actions'] += 1
        
        # Execute control
        controller.u_prev = u
        next_state, next_ref = env.step(u)
        next_obs = controller.construct_observation(next_state, next_ref)
        
        # Store training data (sequence-based)
        if len(controller.sequence_buffer) >= 20:
            seq = np.array(list(controller.sequence_buffer)[-20:])
            controller.training_data.append((seq, u))
        
        # Mode transitions
        if controller.mode == 'collect' and step >= config.pretrain_steps:
            controller.pretrain()
        
        # Fast updates
        if controller.mode == 'transformer' and step % config.fast_update_interval == 0:
            controller.update_model()
        
        # Record data
        if step % config.record_steps == 0:
            controller.results['time'].append(t)
            controller.results['delta'].append(delta)
            controller.results['abs_delta'].append(abs_delta)
            controller.results['control'].append(u)
            controller.results['mode'].append(mode)
        
        # Dwell window statistics
        controller.dwell_counter += 1
        if controller.dwell_counter >= config.dwell_steps:
            mean_error = np.mean(dwell_errors)
            max_error = np.max(dwell_errors)
            
            controller.stats['window_errors'].append(mean_error)
            controller.stats['window_max_errors'].append(max_error)
            
            current_lr = controller.online_optimizer.param_groups[0]['lr']
            controller.stats['learning_rates'].append(current_lr)
            
            # Learning rate scheduling
            if controller.mode == 'transformer':
                controller.scheduler.step(mean_error)
            
            controller.window_count += 1
            print(f"Window {controller.window_count} [{window_start_time:.0f}-{t:.0f}s]: "
                  f"Mean|δ|={mean_error:.6f}, Max|δ|={max_error:.6f}, LR={current_lr:.2e}")
            
            window_start_time = t
            dwell_errors = []
            controller.dwell_counter = 0
        
        # Update state
        state = next_state
        ref = next_ref
        obs = next_obs
        controller.step_count += 1
    
    print(f"\n{'='*70}")
    print("Evaluation Complete")
    print(f"Total steps: {controller.step_count}")
    print(f"{'='*70}\n")
    
    return controller


# ======================================================
# Excel Export
# ======================================================
def export_to_excel(results, stats, config, controller):
    """Export training data to Excel"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'TRANSFORMER_4D_Results_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Sheet 1: Time series data
        df_timeseries = pd.DataFrame({
            'Time (s)': results['time'],
            'Tracking Error δ': results['delta'],
            '|δ|': results['abs_delta'],
            'Control u': results['control'],
            'Mode': results['mode']
        })
        df_timeseries.to_excel(writer, sheet_name='Time Series Data', index=False)
        
        # Sheet 2: Window statistics
        window_data = {
            'Window Number': list(range(1, len(stats['window_errors']) + 1)),
            'Mean |δ|': stats['window_errors'],
            'Max |δ|': stats['window_max_errors'],
            'Learning Rate': stats['learning_rates']
        }
        df_windows = pd.DataFrame(window_data)
        df_windows.to_excel(writer, sheet_name='Window Statistics', index=False)
        
        # Sheet 3: Summary statistics
        abs_delta = np.array(results['abs_delta'])
        raw_delta = np.array(results['delta'])
        
        # Get network info
        net_info = controller.model.get_info()
        
        window_max_abs_delta = stats['window_max_errors']
        
        summary_data = {
            'Metric': [
                'System Order',
                'Lambda (λ)',
                'Total Time (s)',
                'dt (s)',
                'Record Interval (s)',
                'Dwell Time (s)',
                'Total Steps',
                'Recorded Data Points',
                'Total Windows',
                'Overall Mean |δ|',
                'Overall Max |δ|',
                'Overall Min |δ|',
                'Max δ (with sign)',
                'Min δ (with sign)',
                'Max abs(δ) in Window 1',
                'Max abs(δ) in Window 2',
                'Max abs(δ) in Window 3',
                'Max abs(δ) in Window 4',
                'Max abs(δ) in Window 5',
                'Max abs(δ) in Window 6',
                'Final Window Mean |δ|',
                'Best Window Mean |δ|',
                'Best Window Max |δ|',
                'Target Error',
                'Transformer Actions',
                'Safety Interventions',
                'Fallback Activations',
                'Transformer Control Rate (%)',
                'Network Architecture',
                'Total Parameters',
                'Final Learning Rate',
                'PID Kp',
                'PID Kd',
                'PID Ki'
            ],
            'Value': [
                config.system_order,
                config.lambda_c,
                config.max_time,
                config.dt,
                config.record_interval,
                config.dwell_time,
                controller.step_count,
                len(results['time']),
                len(stats['window_errors']),
                np.mean(abs_delta),
                np.max(abs_delta),
                np.min(abs_delta),
                np.max(raw_delta),
                np.min(raw_delta),
                window_max_abs_delta[0] if len(window_max_abs_delta) > 0 else 'N/A',
                window_max_abs_delta[1] if len(window_max_abs_delta) > 1 else 'N/A',
                window_max_abs_delta[2] if len(window_max_abs_delta) > 2 else 'N/A',
                window_max_abs_delta[3] if len(window_max_abs_delta) > 3 else 'N/A',
                window_max_abs_delta[4] if len(window_max_abs_delta) > 4 else 'N/A',
                window_max_abs_delta[5] if len(window_max_abs_delta) > 5 else 'N/A',
                stats['window_errors'][-1] if stats['window_errors'] else 'N/A',
                min(stats['window_errors']) if stats['window_errors'] else 'N/A',
                min(stats['window_max_errors']) if stats['window_max_errors'] else 'N/A',
                config.target_error,
                stats['transformer_actions'],
                stats['safety_interventions'],
                stats['fallback_activations'],
                stats['transformer_actions'] / (controller.step_count - config.pretrain_steps) * 100 if controller.step_count > config.pretrain_steps else 0,
                net_info['architecture'],
                net_info['parameters'],
                stats['learning_rates'][-1] if stats['learning_rates'] else 'N/A',
                config.kp,
                config.kd,
                config.ki
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"\n[Excel Export]")
    print(f"  File: {filename}")
    print(f"  Sheets: Time Series ({len(results['time'])} rows), Windows ({len(stats['window_errors'])}), Summary")
    
    return filename


# ======================================================
# Plotting Functions
# ======================================================
def plot_tracking_error(results, config):
    """Plot standalone tracking error"""
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    t = np.array(results['time'])
    tracking_error = np.array(results['delta'])
    ax.plot(t, tracking_error, 'b-', linewidth=1.5, label='TRANSFORM')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Augmented tracking error', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Window boundary lines
    time_marks = list(range(int(config.dwell_time), int(config.max_time), int(config.dwell_time)))
    for tm in time_marks:
        if tm <= t[-1]:
            ax.axvline(x=tm, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    
    # Target lines
    ax.axhline(y=config.target_error, color='r', linestyle='--', linewidth=1.0, 
               alpha=0.8, label=f'Target (±{config.target_error})')
    ax.axhline(y=-config.target_error, color='r', linestyle='--', linewidth=1.0, alpha=0.8)
    
    ax.set_xlim([0, config.max_time])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(config.dwell_time))
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f'tracking_error_TRANSFORMER_4D_{timestamp}.pdf'
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
    print(f"  Plot: {pdf_filename}")
    plt.show()


def plot_results(results, stats, config):
    """Plot comprehensive results"""
    plt.figure(figsize=(12, 8))
    
    # Error evolution
    plt.subplot(2, 3, 1)
    plt.semilogy(results['time'], results['abs_delta'], alpha=0.7)
    plt.axhline(y=config.target_error, color='r', linestyle='--', label='Target')
    plt.xlabel('Time (s)')
    plt.ylabel('|δ| (log scale)')
    plt.title('Tracking Error (Transformer - 4D)')
    plt.legend()
    plt.grid(True)
    
    # Control signal
    plt.subplot(2, 3, 2)
    plt.plot(results['time'], results['control'], alpha=0.7)
    plt.axhline(y=config.safety_bounds[0], color='orange', linestyle='--', alpha=0.5)
    plt.axhline(y=config.safety_bounds[1], color='orange', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Control u')
    plt.title('Control Signal')
    plt.grid(True)
    
    # Window performance
    plt.subplot(2, 3, 3)
    if stats['window_errors']:
        windows = range(1, len(stats['window_errors'])+1)
        plt.plot(windows, stats['window_errors'], 'o-', label='Mean |δ|')
        plt.plot(windows, stats['window_max_errors'], 's-', label='Max |δ|', alpha=0.7)
        plt.axhline(y=config.target_error, color='r', linestyle='--')
        plt.xlabel('Window Number')
        plt.ylabel('|δ|')
        plt.title('Window Performance')
        plt.legend()
        plt.grid(True)
    
    # Learning rate evolution
    plt.subplot(2, 3, 4)
    if stats['learning_rates']:
        windows = range(1, len(stats['learning_rates'])+1)
        plt.semilogy(windows, stats['learning_rates'], 'o-')
        plt.xlabel('Window Number')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
    
    # Moving average
    plt.subplot(2, 3, 5)
    window_size = 100
    if len(results['delta']) > window_size:
        moving_avg = np.convolve(results['abs_delta'], 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        plt.plot(np.array(results['time'])[window_size-1:], moving_avg)
        plt.axhline(y=config.target_error, color='r', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Moving Avg |δ|')
        plt.title(f'Moving Average (window={window_size})')
        plt.grid(True)
    
    # Error distribution
    plt.subplot(2, 3, 6)
    plt.hist(results['abs_delta'], bins=50, density=True, alpha=0.7)
    plt.axvline(x=config.target_error, color='r', linestyle='--')
    plt.xlabel('|δ|')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f'TRANSFORMER_4D_results_{timestamp}.png'
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"  Plot: {png_filename}")
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("Transformer Controller for 4th-Order Nonlinear System")
    print("="*70)
    
    cfg = SystemConfig(dt=0.01, dwell_time=120.0, max_time=720.0)
    print(f"\n[Configuration]")
    print(f"  System order: {cfg.system_order}")
    print(f"  Lambda (λ): {cfg.lambda_c}")
    print(f"  Dwell time: {cfg.dwell_time}s")
    print(f"  Target error: {cfg.target_error}")
    print(f"  ODE solver: {cfg.ode_method}")
    print(f"  Architecture: Fixed 3-layer Transformer")
    
    controller = evaluate_controller(cfg)
    
    # Export data
    excel_file = export_to_excel(controller.results, controller.stats, cfg, controller)
    
    # Plot results
    plot_tracking_error(controller.results, cfg)
    plot_results(controller.results, controller.stats, cfg)