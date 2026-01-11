import torch
import math
from typing import Optional
import numpy as np

class FinslerAdam(torch.optim.Optimizer):
    """
    FinslerAdam: Geodesic Optimization on Finsler Manifolds for LLMs.
    
    Core Mathematical Concepts:
    --------------------------
    1. Finsler Metric (F):
       Replaces Euclidean norm with a direction-dependent structure.
       F(w, v) = sqrt(g_Adam(w, v)) * L(theta)
    
    2. Dynamic Anisotropy Tensor (A_t):
       Measures local curvature of the information flow.
       A_t = 1 - cos(theta_t) = 1 - <g_t, g_{t-1}> / (|g_t| * |g_{t-1}|)
    
    3. Finsler Scaling Factor (phi):
       Acts as a topological switch and viscosity controller.
       phi(A_t) = 1 + alpha * I(A_t > tau) * A_t
       
       Logic:
       - If A_t < tau (Isotropic region): phi = 1 (Standard Adam).
       - If A_t > tau (Anisotropic region): phi > 1 (Step Size Expansion).
         This "stretches" the loss landscape locally to help the optimizer 
         traverse narrow regions or maintain stability in turbulent high-curvature zones.
    
    Args:
        params (Iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 3e-4).
        betas (Tuple[float, float]): Coefficients for computing running averages 
            of gradient and its square (default: (0.9, 0.999)).
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float): Weight decay coefficient (L2 penalty) (default: 1e-2).
        finsler_factor (float): Alpha, the magnitude of the Finsler distortion (default: 0.15).
        anisotropy_threshold (float): Tau, the topological switch threshold (default: 0.05).
        anisotropy_ema_beta (float): Beta for EMA of anisotropy tensor (default: 0.9).
        log_metrics (bool): Enable logging for geometric analysis (default: False).
    """
    
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, finsler_factor=0.15, anisotropy_threshold=0.05,
                 anisotropy_ema_beta=0.9, log_metrics=False):
        
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                       weight_decay=weight_decay, finsler_factor=finsler_factor,
                       anisotropy_threshold=anisotropy_threshold,
                       anisotropy_ema_beta=anisotropy_ema_beta,
                       log_metrics=log_metrics)
        
        super(FinslerAdam, self).__init__(params, defaults)
        
        # Global logging storage (if enabled)
        if log_metrics:
            self.global_step = 0
            # Using a simple dict for collection to avoid list overhead
            self.metrics_log = {
                'anisotropy': [],
                'finsler_factor': [],
                'effective_lr_ratio': []
            }

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.
            
        Returns:
            loss (Tensor): The value of the loss function at the new point.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # --- 1. Initialization (State Management) ---
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)
                    
                    # Anisotropy buffer initialization (Scalar for simplicity, tensor per layer if needed)
                    state['anisotropy_buffer'] = 0.0
                
                state['step'] += 1
                step_t = state['step']
                
                # --- 2. Adam Standard Update (Momentum & Variance) ---
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # --- 3. Bias Correction ---
                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t
                
                # --- 4. Compute Adam Preconditioning (Riemannian part) ---
                # m_hat = m / bias_correction1
                # v_hat = v / bias_correction2
                
                # Denominator: sqrt(v_hat) + eps
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Preconditioned update vector (Adam update direction)
                u_adam = exp_avg / bias_correction1 / denom
                
                # --- 5. Compute Dynamic Anisotropy Tensor (Finsler part) ---
                prev_grad = state['prev_grad']
                
                current_anisotropy = 0.0
                
                # Only compute if we have history (step > 1)
                if step_t > 1:
                    grad_norm = grad.norm()
                    prev_grad_norm = prev_grad.norm()
                    
                    # Numerical safety
                    if grad_norm > group['eps'] and prev_grad_norm > group['eps']:
                        # Cosine Similarity: <g_t, g_{t-1}> / (||g_t|| * ||g_{t-1}||)
                        # We use flatten() for 1D tensors to ensure consistency
                        cos_sim = torch.sum(
                            grad * prev_grad, 
                            dim=None if grad.dim() == 1 else tuple(range(grad.dim()))
                        ) / (grad_norm * prev_grad_norm)
                        
                        # Clamp for numerical stability
                        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                        
                        # Anisotropy A_t = 1 - cos(theta_t)
                        anisotropy = 1.0 - cos_sim
                        
                        # --- 6. EMA Update for Anisotropy Buffer ---
                        # We smooth the anisotropy to react to global landscape topology, not just step noise
                        state['anisotropy_buffer'] = (
                            group['anisotropy_ema_beta'] * state['anisotropy_buffer'] + 
                            (1 - group['anisotropy_ema_beta']) * anisotropy
                        )
                
                current_anisotropy_buffered = state['anisotropy_buffer']
                
                # --- 7. Compute Finsler Scaling Factor (phi) ---
                # Logic: If anisotropy > threshold, apply distortion. Else phi = 1.0.
                # Formula: phi = 1 + alpha * I(A_t > tau) * A_t
                # Note: We use the EMA'd anisotropy for the decision to reduce jitter.
                
                tau = group['anisotropy_threshold']
                alpha = group['finsler_factor']
                
                # I(A_t > tau) acts as the Indicator function
                if current_anisotropy_buffered > tau:
                    # Apply distortion: 1 + alpha * A_t
                    # Since A_t is in [0, 2], factor is in [1, 1 + 2*alpha]
                    finsler_scale = 1.0 + alpha * current_anisotropy_buffered
                else:
                    # Isotropic region: Standard Euclidean metric (Adam)
                    finsler_scale = 1.0
                
                # --- 8. Geodesic Update (Monge-Ampère Inspired) ---
                # w_{t+1} = w_t - eta * u_adam * phi
                
                # Effective update vector
                u_final = u_adam * finsler_scale
                
                # --- 9. Apply Update ---
                p.data.add_(u_final, alpha=-group['lr'])
                
                # --- 10. Weight Decay (AdamW Style) ---
                if group['weight_decay'] != 0:
                    # p_t = p_t - lr * u - lr * wd * p_t
                    # The `add_` above handles -lr * u_final. We need to add -lr * wd * p
                    # To be efficient: p = p * (1 - lr*wd) + update
                    # Standard AdamW implementation: decay first, then add.
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                
                # --- 11. State Update ---
                state['prev_grad'].copy_(grad)
                
                # --- 12. Scientific Logging (Optional) ---
                if group['log_metrics']:
                    # Log the actual scalar value (using .item() for tensors, direct float for scalars)
                    # This must be CPU-safe.
                    if torch.is_tensor(current_anisotropy_buffered):
                        val_a = current_anisotropy_buffered.item()
                    else:
                        val_a = current_anisotropy_buffered
                    
                    self.metrics_log['anisotropy'].append(val_a)
                    self.metrics_log['finsler_factor'].append(finsler_scale)
                    
                    # Calculate effective LR ratio: How much did we stretch/shrink the step?
                    # Effective Step = Adam_Step * finsler_scale
                    eff_ratio = finsler_scale # Simple scalar or tensor
                    if torch.is_tensor(eff_ratio):
                        eff_ratio = eff_ratio.item()
                    
                    self.metrics_log['effective_lr_ratio'].append(eff_ratio)
                    
                    self.global_step += 1
        
        return loss