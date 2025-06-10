from typing import Optional

import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F


class AccMetric(torchmetrics.Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.correct = 0
        self.total = 0

    def update(
        self, pred: torch.Tensor, target: torch.Tensor, selected_indices: torch.Tensor
    ) -> torch.Tensor:
        if pred.ndim == 3:
            b, n, c = pred.shape
            pred = pred.view(b * n, c)
            target = target.view(-1)
            mask = torch.ones_like(selected_indices)
            mask[selected_indices == -1] = 0
            mask = mask.view(-1)
        self.total += mask.sum(0)
        pred = torch.argmax(pred, dim=1)
        self.correct += ((pred == target) * mask).sum().item()

    def compute(self) -> torch.Tensor:
        return torch.Tensor([self.correct / self.total])


class MSEMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.sum = 0

    def update(
        self, pred: torch.Tensor, target: torch.Tensor, selected_indices: torch.Tensor
    ) -> torch.Tensor:
        if pred.ndim == 3:
            b, n, c = pred.shape
            pred = pred.view(b * n, c)
            target = target.view(b * n, -1)

            mask = torch.ones_like(selected_indices)
            mask[selected_indices == -1] = 0
            mask = mask.view(-1)

        self.total += mask.sum(0)
        self.sum += torch.sum((pred - target) ** 2 * mask.unsqueeze(-1))

    def compute(self) -> torch.Tensor:
        return torch.Tensor([self.sum / self.total])


class EmbAccMetric(torchmetrics.Metric):
    def __init__(
        self, mode: str = "average", batch_contrastive: bool = False, dist_sync_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.mode = mode.lower()
        self.batch_contrastive = batch_contrastive

        # Add state variables
        self.add_state("correct_sc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_cs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        slot_emb: torch.Tensor,
        ctrl_emb: torch.Tensor,
        ctrl_idx: Optional[torch.Tensor] = None,
        slots_idx: Optional[torch.Tensor] = None,
        mask_idx: Optional[torch.Tensor] = None,
    ) -> None:
        assert slot_emb.shape == ctrl_emb.shape
        batch_size, num_slots, dim = slot_emb.size()
        batch_idx = (
            torch.arange(batch_size, device=slot_emb.device)
            .unsqueeze(-1)
            .expand(batch_size, num_slots)
        )
        batch_idx = batch_idx.reshape(-1)

        if ctrl_idx is not None:
            ctrl_indecies_flat = ctrl_idx.reshape(-1).long()
            slots_indecies_flat = slots_idx.reshape(-1).long()
            assert ((ctrl_indecies_flat == -1) == (slots_indecies_flat == -1)).all()
            if mask_idx is not None:
                mask_idx = mask_idx.reshape(-1)
                assert mask_idx == (ctrl_indecies_flat != -1)
            else:
                mask_idx = ctrl_indecies_flat != -1
            # Select ground truth embeddings that match the slot embeddings
            ctrl_emb = ctrl_emb[batch_idx, ctrl_indecies_flat]
            ctrl_emb = ctrl_emb.reshape([batch_size, num_slots, dim])

            slot_emb = slot_emb[batch_idx, slots_indecies_flat]
            slot_emb = slot_emb.reshape([batch_size, num_slots, dim])
        else:
            # if only mask_idx is provided,
            # we assume that the slot and ctrl embeddings are already aligned
            assert mask_idx is not None
            mask_idx = mask_idx.reshape(-1)

        slot_emb = F.normalize(slot_emb, dim=-1, p=2)
        ctrl_emb = F.normalize(ctrl_emb, dim=-1, p=2)
        if not self.batch_contrastive:
            logits_sc = torch.bmm(slot_emb, ctrl_emb.permute(0, 2, 1))
            logits_cs = torch.bmm(ctrl_emb, slot_emb.permute(0, 2, 1))

            labels = torch.arange(num_slots).expand(batch_size, num_slots).to(slot_emb.device)

            # Accuracy calculation
            _, predicted_sc = torch.max(logits_sc, dim=-1)
            _, predicted_cs = torch.max(logits_cs, dim=-1)
            correct_sc = (predicted_sc == labels).float()
            correct_cs = (predicted_cs == labels).float()

            self.correct_sc += (correct_sc.reshape(-1) * mask_idx.float()).sum()
            self.correct_cs += (correct_cs.reshape(-1) * mask_idx.float()).sum()
            self.total += mask_idx.sum()
        else:
            slot_emb = slot_emb.reshape(-1, dim)
            ctrl_emb = ctrl_emb.reshape(-1, dim)
            slot_emb = slot_emb[mask_idx.bool()]
            ctrl_emb = ctrl_emb[mask_idx.bool()]
            logits_sc = slot_emb @ ctrl_emb.T
            logits_cs = ctrl_emb @ slot_emb.T
            batch_size = slot_emb.size(0)
            labels = torch.arange(batch_size, device=slot_emb.device)
            _, predicted_sc = torch.max(logits_sc, dim=-1)
            _, predicted_cs = torch.max(logits_cs, dim=-1)

            correct_sc = (predicted_sc == labels).float()
            correct_cs = (predicted_cs == labels).float()

            self.correct_sc += (correct_sc.reshape(-1)).sum().to(self.correct_sc.device)
            self.correct_cs += (correct_cs.reshape(-1)).sum().to(self.correct_cs.device)
            self.total += mask_idx.sum().to(self.total.device)

    def compute(self) -> torch.Tensor:
        if self.mode == "sc":
            return self.correct_sc / self.total
        elif self.mode == "cs":
            return self.correct_cs / self.total
        elif self.mode == "average":
            return (self.correct_sc + self.correct_cs) / (2 * self.total)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose from 'sc', 'cs', or 'average'.")
