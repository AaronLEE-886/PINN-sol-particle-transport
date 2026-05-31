"""PINN training loops: standard optimization and causal training."""

import torch
import torch.nn as nn
from tqdm import tqdm

from ..physics.params import SOLConfig
from .loss import combined_loss, pde_loss, sheath_loss, upstream_loss


class Trainer:
    """Standard PINN trainer with Adam warmup and L-BFGS refinement."""

    def __init__(self, model, config: SOLConfig, loss_weights=None,
                 lr=1e-3, device="cpu"):
        self.model = model.to(device)
        self.config = config
        self.loss_weights = loss_weights or {"pde": 1.0, "up": 1.0, "sheath": 1.0}
        self.device = device
        self.lr = lr

        self.optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.history = {"total": [], "pde": [], "up": [], "sheath": []}

    def train_adam(self, s_colloc, n_iter=20000, source_fn=None,
                   log_every=1000, grad_clip=1.0):
        """Run the Adam stage."""
        s_colloc = s_colloc.to(self.device)

        pbar = tqdm(range(n_iter), desc="Adam")
        for step in pbar:
            self.optimizer_adam.zero_grad()

            losses = combined_loss(
                self.model, s_colloc, self.config,
                self.loss_weights, source_fn,
            )
            losses["total"].backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer_adam.step()

            for key in self.history:
                if key in losses:
                    self.history[key].append(losses[key].item())

            if step % log_every == 0:
                pbar.set_postfix({
                    "total": f"{losses['total'].item():.2e}",
                    "pde": f"{losses['pde'].item():.2e}",
                    "sheath": f"{losses['sheath'].item():.2e}",
                })

    def train_lbfgs(self, s_colloc, n_iter=500, source_fn=None,
                    log_every=50):
        """Run the L-BFGS stage."""
        s_colloc = s_colloc.to(self.device)
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=0.1,
            max_iter=n_iter,
            max_eval=n_iter * 2,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            losses = combined_loss(
                self.model, s_colloc, self.config,
                self.loss_weights, source_fn,
            )
            losses["total"].backward()
            return losses["total"]

        for step in range(n_iter // log_every):
            loss = optimizer.step(closure)
            if loss is not None:
                print(f"  L-BFGS [{step * log_every}] total={loss.item():.2e}")

    def train(self, s_colloc, n_adam=20000, n_lbfgs=500,
              source_fn=None, log_every=1000):
        """Run the full training pipeline."""
        print("=" * 50)
        print("Phase 1: Adam optimization")
        print("=" * 50)
        self.train_adam(s_colloc, n_adam, source_fn, log_every)

        print("=" * 50)
        print("Phase 2: L-BFGS fine-tuning")
        print("=" * 50)
        self.train_lbfgs(s_colloc, n_lbfgs, source_fn)

        print("Training complete!")


class CausalTrainer(Trainer):
    """Causal PINN trainer with progressively activated spatial intervals."""

    def __init__(self, model, config: SOLConfig, loss_weights=None,
                 lr=1e-3, device="cpu", n_intervals=10, epsilon=0.1,
                 interval_activation_steps=2000):
        super().__init__(model, config, loss_weights, lr, device)
        self.n_intervals = n_intervals
        self.epsilon = epsilon
        self.interval_activation_steps = interval_activation_steps
        self.active_intervals = 1

    def _interval_slice(self, interval_index, n_points, n_per_interval):
        """Return the half-open slice bounds for one interval."""
        start_idx = interval_index * n_per_interval
        end_idx = min((interval_index + 1) * n_per_interval, n_points)
        return start_idx, end_idx

    def train_adam(self, s_colloc, n_iter=20000, source_fn=None,
                   log_every=1000, grad_clip=1.0):
        """Run the Adam stage with causal interval activation."""
        s_colloc = s_colloc.to(self.device)

        s_sorted = torch.sort(s_colloc[:, 0], dim=0).values.unsqueeze(-1)
        n_per_interval = max(1, len(s_sorted) // self.n_intervals)

        interval_loss_acc = torch.zeros(self.n_intervals, device=self.device)
        self.active_intervals = 1

        pbar = tqdm(range(n_iter), desc="Causal-Adam")
        for step in pbar:
            self.optimizer_adam.zero_grad()

            interval_pde_losses = []
            for k in range(self.n_intervals):
                start_idx, end_idx = self._interval_slice(
                    k, len(s_sorted), n_per_interval,
                )
                if start_idx >= end_idx or k >= self.active_intervals:
                    interval_pde_losses.append(torch.tensor(0.0, device=self.device))
                    continue

                s_k = s_sorted[start_idx:end_idx]
                interval_pde_losses.append(
                    pde_loss(self.model, s_k, self.config, source_fn)
                )

            total_pde = torch.tensor(0.0, device=self.device)
            causal_weights = []
            for k, interval_loss in enumerate(interval_pde_losses):
                acc_val = interval_loss_acc[k] + interval_loss.detach()
                interval_loss_acc[k] = acc_val
                w_k = torch.exp(-self.epsilon * acc_val)
                causal_weights.append(w_k)
                total_pde = total_pde + w_k * interval_loss

            weight_sum = torch.stack(causal_weights).sum()
            if weight_sum > 0:
                total_pde = total_pde / weight_sum

            L_up = upstream_loss(self.model, self.config)
            L_sheath = sheath_loss(self.model, self.config)
            total = (
                self.loss_weights.get("pde", 1.0) * total_pde
                + self.loss_weights.get("up", 1.0) * L_up
                + self.loss_weights.get("sheath", 1.0) * L_sheath
            )

            total.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer_adam.step()

            self.history["total"].append(total.item())
            self.history["pde"].append(total_pde.item())
            self.history["up"].append(L_up.item())
            self.history["sheath"].append(L_sheath.item())

            if (step + 1) % self.interval_activation_steps == 0:
                if self.active_intervals < self.n_intervals:
                    self.active_intervals += 1
                    print(
                        f"  Activating interval "
                        f"{self.active_intervals}/{self.n_intervals}"
                    )

            if step % log_every == 0:
                pbar.set_postfix({
                    "total": f"{total.item():.2e}",
                    "active": f"{self.active_intervals}/{self.n_intervals}",
                    "pde": f"{total_pde.item():.2e}",
                })
