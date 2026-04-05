import torch
import torch.nn as nn
from typing import List, Set


class InstructionFingerprintAdapter(nn.Module):
    """Embedding-level adapter: only modifies embeddings of fingerprint instruction tokens.
    Follows the official IF_adapter: trainable_emb + A/B linear layers, zero-initialized."""

    def __init__(self, emb, all_trainable_input_ids, inner_dim=16):
        super().__init__()
        self.orig_emb = emb
        self.all_trainable_input_ids = list(all_trainable_input_ids)
        self.trainable_emb = nn.Embedding(len(self.all_trainable_input_ids), emb.weight.size(1))
        with torch.no_grad():
            self.trainable_emb.weight.copy_(emb.weight[self.all_trainable_input_ids])
        self.A = nn.Linear(emb.weight.size(1), inner_dim)
        self.B = nn.Linear(inner_dim, emb.weight.size(1))
        with torch.no_grad():
            self.A.weight.zero_(); self.A.bias.zero_()
            self.B.weight.zero_(); self.B.bias.zero_()
        self._cast_dtype()

    @property
    def weight(self):
        return self.orig_emb.weight

    @torch.no_grad()
    def _cast_dtype(self):
        dtype = self.orig_emb.weight.dtype
        for param in self.parameters():
            param.data = param.data.to(dtype=dtype)

    def forward(self, input):
        N, L = input.size()
        ids_tensor = torch.tensor(self.all_trainable_input_ids, device=input.device, dtype=input.dtype)
        mask = (input.unsqueeze(-1) == ids_tensor).any(-1)

        if mask.any():
            indices = (input[mask].unsqueeze(-1) == ids_tensor).max(-1).indices.to(self.trainable_emb.weight.device)
            emb_trainable = self.B(self.A(self.trainable_emb(indices))) + self.orig_emb(input[mask])
        emb_orig = self.orig_emb(input[~mask])

        output = torch.empty(N, L, self.orig_emb.weight.size(1),
                             device=input.device, dtype=self.orig_emb.weight.dtype)
        if mask.any():
            output[mask] = emb_trainable
        output[~mask] = emb_orig
        return output

    @torch.no_grad()
    def merge(self):
        self.orig_emb.weight[self.all_trainable_input_ids] = self.trainable_emb.weight


def inject_adapter(model, all_trainable_input_ids, inner_dim=16):
    """Freeze all params, replace embedding with InstructionFingerprintAdapter."""
    for param in model.parameters():
        param.requires_grad = False

    emb_attr_str = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and module is model.get_input_embeddings():
            emb_attr_str = name
    assert emb_attr_str is not None, "Cannot find embedding layer"

    parts = emb_attr_str.split(".")
    parent = model
    for attr in parts[:-1]:
        parent = getattr(parent, attr)

    emb = model.get_input_embeddings()
    adapter = InstructionFingerprintAdapter(emb, list(all_trainable_input_ids), inner_dim=inner_dim)
    adapter = adapter.to(device=emb.weight.device, dtype=emb.weight.dtype)
    setattr(parent, parts[-1], adapter)

    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    model.get_input_embeddings().orig_emb.weight.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Injected adapter: {trainable:,} trainable params (inner_dim={inner_dim})", flush=True)
    return model


def unwrap_adapter(model):
    """Merge trainable embeddings back and restore original embedding layer."""
    adapter = model.get_input_embeddings()
    assert isinstance(adapter, InstructionFingerprintAdapter)
    adapter.merge()

    emb_attr_str = None
    for name, module in model.named_modules():
        if isinstance(module, InstructionFingerprintAdapter):
            emb_attr_str = name

    parts = emb_attr_str.split(".")
    parent = model
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, parts[-1], adapter.orig_emb)
    return model, adapter
