"""Isolated follow-up models and loss controls built on frozen version-1 code."""
import torch
from torch import nn
from torch.nn import functional as F

from base_models import HybridDR as OriginalModel
from base_models import objective, probabilities


class ScaledExchange(nn.Module):
    def __init__(self, block, initial_gain):
        super().__init__()
        self.block = block
        self.gain = nn.Parameter(torch.tensor(float(initial_gain)))

    def forward(self, query, key, qpos, kpos):
        correction = self.block(query, key, qpos, kpos) - query
        return query + self.gain * correction


class HybridDR(OriginalModel):
    def __init__(self, config, recipe, initialize=True):
        super().__init__(config, recipe, initialize)
        if 'residual_gain_init' in recipe:
            if self.arch != 'spatial':
                raise ValueError('Residual scaling requires spatial exchange')
            self.vexchange = ScaledExchange(self.vexchange, recipe['residual_gain_init'])
            if self.cexchange is not None:
                self.cexchange = ScaledExchange(self.cexchange, recipe['residual_gain_init'])


def class_weights(counts, tempered=False):
    counts = counts.float().clamp_min(1)
    if tempered:
        weights = counts.rsqrt()
        return weights / (weights * counts / counts.sum()).sum()
    weights = counts.reciprocal()
    return weights / weights.mean()


def training_objective(logits, labels, recipe, epoch, total_epochs, counts, smoothing):
    kind = recipe['loss']
    if kind not in ('blend_uniform', 'blend_tempered'):
        return objective(logits, labels, kind, epoch, total_epochs,
                         class_weights(counts), smoothing)
    log_probs = logits.float().log_softmax(-1)
    selected = log_probs.gather(1, labels[:, None]).squeeze(1)
    weights = class_weights(counts, tempered=True)[labels] if kind == 'blend_tempered' else torch.ones_like(selected)
    ramp_epochs = recipe.get('blend_ramp_epochs', 10)
    if ramp_epochs <= 0:
        raise ValueError('Blend ramp must be positive')
    blend = recipe.get('blend_max', .5) * min(epoch / ramp_epochs, 1.)
    if not 0 <= blend <= 1:
        raise ValueError('Blend coefficient must be in [0,1]')
    focal = -weights * (1 - selected.exp()).pow(recipe.get('focal_gamma', 1.)) * selected
    hard = (1 - blend) * -selected + blend * focal
    # The uniform smoothing contribution remains independent of modulation.
    return ((1 - smoothing) * hard - smoothing * log_probs.mean(-1)).mean()
