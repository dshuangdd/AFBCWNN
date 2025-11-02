# RBF Adaptive Controller - Simplified Data Version
# Features:
# - Two network configurations: RBF-256 (256 centers) and RBF-625 (625 centers)
# - Optional disturbance testing (toggle via configuration)
# - High-precision computation (480,000 points) for accuracy
# - Downsampled Excel output (10,000 points) for visualization
# - Weight update: ACWNN-style 

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from torch.optim import Adam
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os

class RBFNetwork(nn.Module):
    """
    Radial Basis Function Network for 4D system
    
    Network configurations:
    - RBF-256: 256 centers (4^4 grid) - faster, lower capacity
    - RBF-625: 625 centers (5^4 grid) - slower, higher capacity
    """
    def __init__(self, input_dim=4, num_centers=256, center_range=None):
        super(RBFNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        
        if center_range is None:
            center_range = [
                [-1.0, 2.0],   # x1
                [-2.0, 2.0],   # x2
                [-2.0, 2.0],   # x3
                [-3.0, 3.0]    # x4
            ]
        
        self.centers = self._generate_centers(center_range)
        self.widths = self._compute_widths()
        self.output_weights = nn.Parameter(torch.zeros(self.centers.shape[0]))
        nn.init.uniform_(self.output_weights, -0.01, 0.01)
        
        print(f"  Actual RBF centers: {self.centers.shape[0]}")
        
    def _generate_centers(self, center_range):
        """Generate RBF centers on a uniform grid"""
        points_per_dim = int(np.round(self.num_centers ** (1.0 / self.input_dim)))
        print(f"  Grid points per dimension: {points_per_dim}")
        
        grids = []
        for i in range(self.input_dim):
            grids.append(np.linspace(center_range[i][0], center_range[i][1], points_per_dim))
        
        mesh = np.meshgrid(*grids, indexing='ij')
        centers = np.stack([m.flatten() for m in mesh], axis=1)
        
        return torch.FloatTensor(centers)
    
    def _compute_widths(self):
        """Compute RBF widths based on nearest neighbor distances"""
        num_samples = min(100, self.centers.shape[0])
        indices = torch.randperm(self.centers.shape[0])[:num_samples]
        sampled_centers = self.centers[indices]
        
        dists = torch.cdist(sampled_centers, self.centers)
        dists_sorted, _ = torch.sort(dists, dim=1)
        nearest_dists = dists_sorted[:, 1]
        
        avg_nearest_dist = nearest_dists.mean().item()
        width = avg_nearest_dist * 1.5
        width = np.clip(width, 0.5, 3.0)
        
        widths = torch.full((self.centers.shape[0],), width)
        print(f"  RBF width: {width:.4f}")
        
        return widths
    
    def basis_functions(self, x):
        """Compute RBF basis functions"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        dists = torch.cdist(x, self.centers)
        widths = self.widths.unsqueeze(0)
        rbf_activations = torch.exp(-dists**2 / (2 * widths**2))
        
        if x.size(0) == 1:
            return rbf_activations.squeeze(0)
        return rbf_activations
    
    def forward(self, x):
        """Forward pass through RBF network"""
        phi = self.basis_functions(x)
        output = torch.matmul(phi, self.output_weights)
        
        if phi.dim() == 1:
            return output
        return output


class FourDimensionalSystem:
    """
    4D nonlinear system with optional disturbance
    
    Disturbance options:
    - Disabled: Standard system without external perturbation
    - Enabled: Sinusoidal disturbance d(t) = magnitude * sin(frequency * t)
      Applied within specified time window [start_time, end_time]
    """
    def __init__(self, dt=0.1, disturbance_config=None):
        self.dt = dt
        self.state = np.zeros(4)
        
        # Disturbance configuration
        if disturbance_config is None:
            disturbance_config = {'enabled': False}
        
        self.disturbance_enabled = disturbance_config.get('enabled', False)
        self.disturbance_start_time = disturbance_config.get('start_time', 240.0)
        self.disturbance_end_time = disturbance_config.get('end_time', 360.0)
        self.disturbance_magnitude = disturbance_config.get('magnitude', 0.5)
        self.disturbance_frequency = disturbance_config.get('frequency', 5.0)
        
        if self.disturbance_enabled:
            print(f"\n[Disturbance Enabled]")
            print(f"  Time window: [{self.disturbance_start_time}, {self.disturbance_end_time}]s")
            print(f"  Magnitude: {self.disturbance_magnitude}")
            print(f"  Frequency: {self.disturbance_frequency} rad/s")
            print(f"  Form: d(t) = {self.disturbance_magnitude} * sin({self.disturbance_frequency}*t)")
        
    def reset(self, initial_state=None):
        """Reset system to initial state"""
        if initial_state is None:
            self.state = np.array([0.5, 0.0, 0.0, 0.0])
        else:
            self.state = np.array(initial_state)
        return self.state.copy()
    
    def nonlinear_function(self, x):
        """System nonlinearity"""
        f = -(x[0] + 2*x[1] + 2*x[2] + 2*x[3]) + \
            (1 - (x[0] + 2*x[1] + x[2])**2) * (x[1] + 2*x[2] + x[3])
        return f
    
    def compute_disturbance(self, t):
        """Compute disturbance at current time"""
        if not self.disturbance_enabled:
            return 0.0
        
        if self.disturbance_start_time <= t < self.disturbance_end_time:
            return self.disturbance_magnitude * np.sin(self.disturbance_frequency * t)
        
        return 0.0
    
    def dynamics(self, x, u, t):
        """System dynamics with disturbance"""
        x_dot = np.zeros(4)
        x_dot[0] = x[1]
        x_dot[1] = x[2]
        x_dot[2] = x[3]
        
        d = self.compute_disturbance(t)
        x_dot[3] = self.nonlinear_function(x) + u + d
        
        return x_dot
    
    def step_ode(self, u, current_time):
        """Advance system by one time step using ODE solver"""
        def dynamics_func(x, t):
            return self.dynamics(x, u, current_time)
        
        t_span = [0, self.dt]
        sol = odeint(dynamics_func, self.state, t_span)
        self.state = sol[-1]
        
        return self.state.copy()


def reference_trajectory(t):
    """Reference trajectory"""
    return 0.5 * np.sin(t)**2


def reference_derivatives(t):
    """Reference trajectory and its derivatives"""
    yr = 0.5 * np.sin(t)**2
    yr_dot = np.sin(t) * np.cos(t)
    yr_ddot = np.cos(2*t)
    yr_dddot = -2 * np.sin(2*t)
    return yr, yr_dot, yr_ddot, yr_dddot


def compute_augmented_error(x, t, lam):
    """Compute augmented tracking error"""
    yr, yr_dot, yr_ddot, yr_dddot = reference_derivatives(t)
    
    delta_x = np.array([
        x[0] - yr,
        x[1] - yr_dot,
        x[2] - yr_ddot,
        x[3] - yr_dddot
    ])
    
    lambda_vec = np.array([lam**3, 3*lam**2, 3*lam, 1])
    delta = np.dot(lambda_vec, delta_x)
    
    return delta, delta_x


def control_law(x, t, f_hat, beta, lam):
    """Control law computation"""
    yr, yr_dot, yr_ddot, yr_dddot = reference_derivatives(t)
    delta, delta_x = compute_augmented_error(x, t, lam)
    
    lambda_vec = np.array([0, lam**3, 3*lam**2, 3*lam])
    g = -yr_dddot + np.dot(lambda_vec, delta_x)
    
    u = -beta * delta - f_hat + g
    
    return u, delta


def downsample_data(data_array, target_points=10000):
    """Downsample data to target number of points"""
    if len(data_array) <= target_points:
        return data_array
    
    indices = np.linspace(0, len(data_array) - 1, target_points, dtype=int)
    return data_array[indices]


def train_rbf_controller(num_centers=256, network_name="RBF-256", 
                        disturbance_config=None):
    """
    Train RBF controller
    
    Parameters:
    - num_centers: Number of RBF centers (256 or 625 recommended)
    - network_name: Network identifier for output files
    - disturbance_config: Disturbance configuration dict (None for no disturbance)
    """
    
    # Simulation parameters
    dt = 0.002
    T_total = 960
    num_steps = int(T_total / dt)
    
    beta = 10.0
    lam = 2.0
    
    learning_rate = 0.0001
    betas = (0.9, 0.999)
    eps = 1e-8
    
    use_lr_decay = True
    lr_decay_rate = 0.9999
    
    # Initialize system
    system = FourDimensionalSystem(dt=dt, disturbance_config=disturbance_config)
    x = system.reset()
    
    print(f"\n[Initializing {network_name}]")
    rbf_net = RBFNetwork(input_dim=4, num_centers=num_centers)
    
    optimizer = Adam(rbf_net.parameters(), 
                     lr=learning_rate,
                     betas=betas,
                     eps=eps)
    
    if use_lr_decay:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)
    
    # Full-precision data recording
    time_history = []
    error_history = []
    weights_norm_history = []
    
    print(f"\n{'='*60}")
    print(f"Training {network_name}")
    print(f"{'='*60}")
    print(f"Update method: ACWNN-style (augmented error only)")
    print(f"Control params: β={beta}, λ={lam}")
    print(f"Duration: {T_total}s, Step: {dt}s")
    print(f"Total steps: {num_steps} (full precision)")
    print(f"Excel points: 10000 (downsampled)")
    print("-" * 60)
    
    # Initialize ADAM state
    adam_m = torch.zeros_like(rbf_net.output_weights.data)
    adam_v = torch.zeros_like(rbf_net.output_weights.data)
    adam_t = 0
    
    # Main training loop
    for step in range(num_steps):
        t = step * dt
        
        # Progress reporting
        if step % 10000 == 0 or step == num_steps - 1:
            progress = (step + 1) / num_steps * 100
            current_error = error_history[-1] if error_history else 0.0
            current_weights = weights_norm_history[-1] if weights_norm_history else 0.0
            print(f"Progress: {progress:6.2f}% | t: {t:7.2f}s | "
                  f"error: {current_error:8.5f} | "
                  f"||w||: {current_weights:8.4f}")
        
        # Convert to torch tensor
        x_tensor = torch.FloatTensor(x)
        
        # RBF estimation
        f_hat_tensor = rbf_net(x_tensor)
        f_hat = f_hat_tensor.item()
        
        # Control law
        u, delta = control_law(x, t, f_hat, beta, lam)
        
        # Get basis functions
        phi = rbf_net.basis_functions(x_tensor)
        
        # ACWNN method: gradient using augmented error only
        # Key: Does not use f_true (unknown in practice), only uses delta
        grad = delta * phi
        
        # Manual ADAM update with control-theoretic gradient
        adam_t += 1
        adam_m = betas[0] * adam_m + (1 - betas[0]) * grad
        adam_v = betas[1] * adam_v + (1 - betas[1]) * (grad ** 2)
        
        m_hat = adam_m / (1 - betas[0]**adam_t)
        v_hat = adam_v / (1 - betas[1]**adam_t)
        
        current_lr = learning_rate if not use_lr_decay else optimizer.param_groups[0]['lr']
        
        with torch.no_grad():
            rbf_net.output_weights.data -= current_lr * m_hat / (torch.sqrt(v_hat) + eps)
        
        # Learning rate decay
        if use_lr_decay:
            scheduler.step()
        
        # System step
        x = system.step_ode(u, t)
        
        # Record data
        time_history.append(t)
        error_history.append(delta)
        weights_norm_history.append(torch.norm(rbf_net.output_weights.data).item())
    
    print("-" * 60)
    print(f"Training complete!\n")
    
    # Convert to numpy arrays
    time_history = np.array(time_history)
    error_history = np.array(error_history)
    weights_norm_history = np.array(weights_norm_history)
    
    # Performance metrics (full precision)
    error_abs = np.abs(error_history)
    max_error = np.max(error_abs)
    final_error = np.mean(error_abs[-5000:])
    
    # MATEDT statistics (every 120s)
    window_size = int(120 / dt)
    num_windows = len(error_abs) // window_size
    matedt = []
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window_max = np.max(error_abs[start_idx:end_idx])
        matedt.append(window_max)
    
    print(f"[Performance Metrics] (based on {len(error_history)} points)")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Final error mean: {final_error:.6f}")
    print(f"  Target: 0.12")
    
    if final_error < 0.12:
        print(f"  ✓ Converged to target!\n")
    else:
        print(f"  ✗ Not converged\n")
    
    # Downsample for Excel
    target_excel_points = 10000
    time_excel = downsample_data(time_history, target_excel_points)
    error_excel = downsample_data(error_history, target_excel_points)
    weights_norm_excel = downsample_data(weights_norm_history, target_excel_points)
    
    print(f"Data downsampled: {len(time_history)} → {len(time_excel)} points\n")
    
    # Return results
    results = {
        'network_name': network_name,
        'num_centers': rbf_net.centers.shape[0],
        
        # Excel data (downsampled)
        'time_excel': time_excel,
        'error_excel': error_excel,
        'weights_norm_excel': weights_norm_excel,
        
        # Full data (for accurate computation)
        'time_full': time_history,
        'error_full': error_history,
        'weights_norm_full': weights_norm_history,
        
        # Performance metrics
        'max_error_overall': max_error,
        'final_error_mean': final_error,
        'matedt': matedt,
        'disturbance_config': system.disturbance_enabled,
    }
    
    return results


def save_single_network_excel(results, filename):
    """Save network data to Excel (simplified version)"""
    
    # Sheet1: Main data (downsampled, for plotting)
    df_main = pd.DataFrame({
        'time': results['time_excel'],
        'tracking_error': results['error_excel'],
        'weights_norm': results['weights_norm_excel']
    })
    
    # Sheet2: Max error statistics (computed from full data)
    error_abs = np.abs(results['error_full'])
    df_max_delta = pd.DataFrame({
        'Max|tracking_error|': [np.max(error_abs)],
        'Mean|tracking_error|': [np.mean(error_abs)],
        'Final|tracking_error|_mean': [results['final_error_mean']],
        'Target': [0.12],
        'Data_points_used': [len(results['error_full'])]
    })
    
    # Sheet3: MATEDT (computed from full data)
    df_matedt = pd.DataFrame({
        'Interval': [f"[{i*120}, {(i+1)*120}]" for i in range(len(results['matedt']))],
        'Max|tracking_error|': results['matedt']
    })
    
    # Sheet4: Summary
    df_summary = pd.DataFrame({
        'Metric': [
            'Network Name',
            'Number of Centers',
            'Max Error Overall',
            'Final Error Mean',
            'Target Accuracy',
            'Converged',
            'Final Weight Norm',
            'Disturbance Enabled',
            'Full Data Points',
            'Excel Data Points'
        ],
        'Value': [
            results['network_name'],
            results['num_centers'],
            f"{results['max_error_overall']:.6f}",
            f"{results['final_error_mean']:.6f}",
            '0.12',
            '✓ Yes' if results['final_error_mean'] < 0.12 else '✗ No',
            f"{results['weights_norm_excel'][-1]:.4f}",
            '✓ Yes' if results['disturbance_config'] else '✗ No',
            len(results['error_full']),
            len(results['error_excel'])
        ]
    })
    
    # Write to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Sheet1', index=False)
        df_max_delta.to_excel(writer, sheet_name='Max_delta', index=False)
        df_matedt.to_excel(writer, sheet_name='MATEDT', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✓ Excel saved: {filename}")
    print(f"  - Sheet1: 3 columns × {len(df_main)} rows")
    print(f"  - Metrics based on {len(results['error_full'])} full data points")


def create_comparison_plots(results_256, results_625, output_dir):
    """Create comparison plots (using full precision data)"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    has_disturbance = results_256['disturbance_config'] or results_625['disturbance_config']
    dist_suffix = "_disturbed" if has_disturbance else ""
    
    pdf_filename = os.path.join(output_dir, f'RBF_Comparison_256_vs_625{dist_suffix}.pdf')
    
    with PdfPages(pdf_filename) as pdf:
        color_256 = '#2E86AB'
        color_625 = '#A23B72'
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Tracking error
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(results_256['time_full'], results_256['error_full'], 
                 color=color_256, linewidth=1.5, label='RBF-256', alpha=0.8)
        ax1.plot(results_625['time_full'], results_625['error_full'], 
                 color=color_625, linewidth=1.5, label='RBF-625', alpha=0.8)
        ax1.axhline(y=0.12, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target=0.12')
        ax1.axhline(y=-0.12, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Tracking Error δ(t)', fontsize=11)
        ax1.set_title('Tracking Error', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Absolute error (log scale)
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(results_256['time_full'], np.abs(results_256['error_full']), 
                 color=color_256, linewidth=1.5, label='RBF-256', alpha=0.8)
        ax2.plot(results_625['time_full'], np.abs(results_625['error_full']), 
                 color=color_625, linewidth=1.5, label='RBF-625', alpha=0.8)
        ax2.axhline(y=0.12, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target=0.12')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('|δ(t)| (log)', fontsize=11)
        ax2.set_title('Absolute Error (Log Scale)', fontweight='bold', fontsize=12)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')
        
        # 3. Weight norm evolution
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(results_256['time_full'], results_256['weights_norm_full'], 
                 color=color_256, linewidth=1.5, 
                 label=f"RBF-256 ({results_256['num_centers']} centers)")
        ax3.plot(results_625['time_full'], results_625['weights_norm_full'], 
                 color=color_625, linewidth=1.5, 
                 label=f"RBF-625 ({results_625['num_centers']} centers)")
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('||w||₂', fontsize=11)
        ax3.set_title('Weight Norm Evolution', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MATEDT comparison
        ax4 = plt.subplot(2, 3, 4)
        x_pos = np.arange(len(results_256['matedt']))
        width = 0.35
        ax4.bar(x_pos - width/2, results_256['matedt'], width, 
                label='RBF-256', color=color_256, alpha=0.8)
        ax4.bar(x_pos + width/2, results_625['matedt'], width, 
                label='RBF-625', color=color_625, alpha=0.8)
        ax4.axhline(y=0.12, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target=0.12')
        ax4.set_xlabel('Interval', fontsize=11)
        ax4.set_ylabel('Max|δ|', fontsize=11)
        ax4.set_title('MATEDT Comparison', fontweight='bold', fontsize=12)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"I{i+1}" for i in range(len(results_256['matedt']))], fontsize=9)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Error convergence (smoothed)
        ax5 = plt.subplot(2, 3, 5)
        window = 5000
        error_256_ma = np.convolve(np.abs(results_256['error_full']), 
                                   np.ones(window)/window, mode='valid')
        error_625_ma = np.convolve(np.abs(results_625['error_full']), 
                                   np.ones(window)/window, mode='valid')
        time_ma = results_256['time_full'][window-1:]
        
        ax5.plot(time_ma, error_256_ma, color=color_256, linewidth=2, 
                label='RBF-256 (Moving Avg)', alpha=0.8)
        ax5.plot(time_ma, error_625_ma, color=color_625, linewidth=2, 
                label='RBF-625 (Moving Avg)', alpha=0.8)
        ax5.axhline(y=0.12, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax5.set_xlabel('Time (s)', fontsize=11)
        ax5.set_ylabel('|δ(t)| Moving Average', fontsize=11)
        ax5.set_title('Error Convergence (Smoothed)', fontweight='bold', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance metrics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        metrics_data = [
            ['Metric', 'RBF-256', 'RBF-625'],
            ['Centers', f"{results_256['num_centers']}", f"{results_625['num_centers']}"],
            ['Max|δ|', f"{results_256['max_error_overall']:.4f}", 
             f"{results_625['max_error_overall']:.4f}"],
            ['Final Mean|δ|', f"{results_256['final_error_mean']:.4f}", 
             f"{results_625['final_error_mean']:.4f}"],
            ['Target', '0.12', '0.12'],
            ['Converged', 
             '✓' if results_256['final_error_mean'] < 0.12 else '✗',
             '✓' if results_625['final_error_mean'] < 0.12 else '✗'],
            ['Final ||w||', f"{results_256['weights_norm_full'][-1]:.2f}",
             f"{results_625['weights_norm_full'][-1]:.2f}"]
        ]
        
        table = ax6.table(cellText=metrics_data, loc='center', cellLoc='center',
                         colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(3):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
        
        ax6.set_title('Performance Metrics', fontweight='bold', fontsize=12, pad=20)
        
        title_suffix = " (With Disturbance)" if has_disturbance else ""
        plt.suptitle(f'RBF-256 vs RBF-625 Performance Comparison{title_suffix}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Comparison plots saved: {pdf_filename}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RBF Adaptive Controller - Simplified Data Version")
    print("="*70)
    print("Features:")
    print("  • Two network configurations: RBF-256 and RBF-625")
    print("  • Optional disturbance testing (toggle via configuration)")
    print("  • Full precision computation (480,000 points)")
    print("  • Downsampled Excel output (10,000 points)")
    print("  • Target tracking error: 0.12")
    print("="*70)
    
    # ========== Disturbance Configuration ==========
    # Users can toggle disturbance testing on/off
    
    # Option A: No disturbance (baseline)
    # disturbance_config = {
    #     'enabled': False,
    # }
    
    
    # Option B: Strong disturbance (current setting)
    disturbance_config = {
         'enabled': True,
         'start_time': 120.0,
         'end_time': 240000.0,
         'magnitude': 6.0,
         'frequency': 4.0
     }
    
    print("\n[Configuration]")
    print(f"  Disturbance: {'Enabled' if disturbance_config['enabled'] else 'Disabled'}")
    if disturbance_config['enabled']:
        print(f"  Time window: [{disturbance_config.get('start_time', 0)}, "
              f"{disturbance_config.get('end_time', 0)}]s")
        print(f"  Magnitude: {disturbance_config.get('magnitude', 0)}")
    print("="*70)
    
    # Create output directory
    output_dir = 'rbf_results_simplified'
    os.makedirs(output_dir, exist_ok=True)
    
    # Train RBF-256
    print("\n" + "="*70)
    print("Training RBF-256 Network")
    print("="*70)
    results_256 = train_rbf_controller(
        num_centers=256, 
        network_name="RBF-256",
        disturbance_config=disturbance_config
    )
    
    # Save RBF-256 data
    excel_256 = os.path.join(output_dir, 'RBF_4D_ADAM_256nodes_disturbed.xlsx')
    save_single_network_excel(results_256, excel_256)
    
    # Train RBF-625
    print("\n" + "="*70)
    print("Training RBF-625 Network")
    print("="*70)
    results_625 = train_rbf_controller(
        num_centers=625, 
        network_name="RBF-625",
        disturbance_config=disturbance_config
    )
    
    # Save RBF-625 data
    excel_625 = os.path.join(output_dir, 'RBF_4D_ADAM_625nodes_disturbed.xlsx')
    save_single_network_excel(results_625, excel_625)
    
    # Create comparison plots
    create_comparison_plots(results_256, results_625, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("Training Complete - Files Saved")
    print("="*70)
    print(f"Output directory: {output_dir}/")
    print(f"\nRBF-256 Excel: RBF_4D_ADAM_256nodes_disturbed.xlsx")
    print(f"RBF-625 Excel: RBF_4D_ADAM_625nodes_disturbed.xlsx")
    print(f"Comparison plots: RBF_Comparison_256_vs_625.pdf")
    
    print("\n" + "="*70)
    print("Performance Comparison (480,000 data points)")
    print("="*70)
    print(f"{'Metric':<35} {'RBF-256':<20} {'RBF-625':<20}")
    print("-" * 75)
    print(f"{'RBF Centers':<35} {results_256['num_centers']:<20} {results_625['num_centers']:<20}")
    print(f"{'Max Error':<35} {results_256['max_error_overall']:<20.6f} {results_625['max_error_overall']:<20.6f}")
    print(f"{'Final Mean Error':<35} {results_256['final_error_mean']:<20.6f} {results_625['final_error_mean']:<20.6f}")
    print(f"{'Target':<35} {'0.12':<20} {'0.12':<20}")
    print(f"{'Converged':<35} "
          f"{'✓ Yes' if results_256['final_error_mean'] < 0.12 else '✗ No':<20} "
          f"{'✓ Yes' if results_625['final_error_mean'] < 0.12 else '✗ No':<20}")
    print(f"{'Final Weight Norm':<35} {results_256['weights_norm_full'][-1]:<20.4f} {results_625['weights_norm_full'][-1]:<20.4f}")
    print("="*70)
    print("\n✅ All files saved successfully!")