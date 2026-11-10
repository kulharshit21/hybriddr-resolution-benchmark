"""Matched encoders and genuine spatial cross-attention with gradient flow."""
import math

import timm
import torch
from torch import nn
from torch.nn import functional as F


class TinyEncoder(nn.Module):
    """Smoke-test-only encoder: never included in research result tables."""
    def __init__(self, width=32):
        super().__init__()
        self.num_features = width
        self.net = nn.Sequential(nn.Conv2d(3, width, 5, stride=4, padding=2), nn.GELU(),
                                 nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU())

    def forward_features(self, x):
        return self.net(x)


def coordinates(h, w, device):
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=device),
                          torch.linspace(0, 1, w, device=device), indexing='ij')
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


class Exchange(nn.Module):
    def __init__(self, width, heads, spatial_bias):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.position = nn.Linear(2, width)
        self.raw_scale = nn.Parameter(torch.tensor(0.0)) if spatial_bias else None
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, query, key, qpos, kpos):
        mask = None
        if self.raw_scale is not None:
            mask = -F.softplus(self.raw_scale) * torch.cdist(qpos.float(), kpos.float()).square()
            mask = mask.to(query.dtype)
        q = self.query_norm(query + self.position(qpos).to(query.dtype))
        k = self.key_norm(key + self.position(kpos).to(key.dtype))
        refined, _ = self.attention(q, k, key, attn_mask=mask, need_weights=False)
        output = query + refined
        return output + self.ff(output)


class OrderedHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Linear(width, 1)
        self.base = nn.Parameter(torch.tensor(-1.5))
        self.increments = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        cuts = torch.cat((self.base.reshape(1), self.base + F.softplus(self.increments).cumsum(0)))
        return self.score(x) - cuts


def probabilities(logits, ordinal=False, temperature=1.0):
    if not ordinal:
        return (logits.float() / temperature).softmax(-1)
    cumulative = (logits.float() / temperature).sigmoid()
    return torch.cat((1 - cumulative[:, :1], cumulative[:, :-1] - cumulative[:, 1:], cumulative[:, -1:]), dim=1)


class HybridDR(nn.Module):
    def __init__(self, config, recipe, initialize=True):
        super().__init__()
        self.recipe = recipe
        self.arch = recipe['architecture']
        self.tiny = config.get('tiny', False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.vit = None
        self.cnn = None
        if self.arch != 'cnn':
            self.vit = TinyEncoder() if self.tiny else timm.create_model(config['vit'], pretrained=config['pretrained'] and initialize,
                                                                        num_classes=0, img_size=config['image_size'])
        if self.arch != 'vit':
            self.cnn = TinyEncoder() if self.tiny else timm.create_model(config['cnn'], pretrained=config['pretrained'] and initialize, num_classes=0)
        for encoder in (self.vit, self.cnn):
            if encoder is not None and not self.tiny:
                cfg = timm.data.resolve_model_data_config(encoder)
                if tuple(cfg['mean']) != (.485, .456, .406) or tuple(cfg['std']) != (.229, .224, .225):
                    raise ValueError('Encoder requires different normalization; configure branch preprocessing explicitly')
                if config.get('gradient_checkpointing') and hasattr(encoder, 'set_grad_checkpointing'):
                    encoder.set_grad_checkpointing(True)
        width = config['fusion_width']
        self.vproj = nn.Linear(self.vit.num_features, width) if self.vit else None
        self.cproj = nn.Linear(self.cnn.num_features, width) if self.cnn else None
        dual = self.arch not in ('cnn', 'vit')
        if self.arch == 'spatial':
            self.vexchange = Exchange(width, config['fusion_heads'], recipe.get('spatial_bias', False))
            self.cexchange = Exchange(width, config['fusion_heads'], recipe.get('spatial_bias', False)) if recipe.get('bidirectional', True) else None
        if self.arch == 'gated':
            self.gate = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.LayerNorm(width * (2 if dual else 1)),
                                    nn.Linear(width * (2 if dual else 1), width), nn.GELU(), nn.Dropout(.2))
        self.head = OrderedHead(width) if recipe['loss'] == 'ordinal' else nn.Linear(width, 5)

    def tokens(self, encoder, x):
        feats = encoder.forward_features(x)
        if feats.ndim == 4:
            h, w = feats.shape[-2:]
            feats = feats.flatten(2).transpose(1, 2)
        else:
            feats = feats[:, getattr(encoder, 'num_prefix_tokens', 1):]
            h = w = math.isqrt(feats.shape[1])
            if h * w != feats.shape[1]:
                raise ValueError('Expected square spatial token grid')
        return feats, coordinates(h, w, x.device)

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.vit is not None:
            v, vp = self.tokens(self.vit, x)
            v = self.vproj(v)
        if self.cnn is not None:
            c, cp = self.tokens(self.cnn, x)
            c = self.cproj(c)
        if self.arch == 'vit':
            combined = v.mean(1)
        elif self.arch == 'cnn':
            combined = c.mean(1)
        else:
            if self.arch == 'spatial':
                new_v = self.vexchange(v, c, vp, cp)
                c = self.cexchange(c, v, cp, vp) if self.cexchange else c
                v = new_v
            v, c = v.mean(1), c.mean(1)
            if self.arch == 'gated':
                g = self.gate(torch.cat((v, c), -1))
                v, c = v * g, c * (1 - g)
            combined = torch.cat((v, c), -1)
        return self.head(self.fusion(combined))


def objective(logits, labels, kind, epoch, total_epochs, alpha, smoothing=.1):
    if kind == 'ordinal':
        targets = (labels[:, None] > torch.arange(4, device=labels.device)).float()
        return F.binary_cross_entropy_with_logits(logits.float(), targets)
    if kind == 'ce':
        return F.cross_entropy(logits.float(), labels, label_smoothing=smoothing)
    logp = logits.float().log_softmax(-1)
    selected = logp.gather(1, labels[:, None]).squeeze(1)
    delta = logits.new_tensor([0, 0, 0, .5, .3]).float()
    if kind == 'pfl':
        ramp = min(epoch / 20, 1)
    elif kind == 'reverse_pfl':
        ramp = min((total_epochs - 1 - epoch) / 20, 1)
    elif kind == 'matched_focal':
        ramp = sum(min(e / 20, 1) for e in range(total_epochs)) / total_epochs
    else:
        ramp = 0
    weights = alpha * (1 + delta * ramp)
    smooth_ce = -(1 - smoothing) * selected - smoothing * logp.mean(-1)
    return (weights[labels] * (1 - selected.exp()).pow(2.5) * smooth_ce).mean()
