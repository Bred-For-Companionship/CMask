from dataclasses import dataclass
from transformers.modeling_outputs import MaskedLMOutput
import torch
from typing import Tuple

@dataclass
''''output value matrices and attns for cross attn layers to calc reward'''
class MaskedLMOutputCrossAttn(MaskedLMOutput):
    attentions: Tuple[torch.FloatTensor] = None
    value_matrices: Tuple[torch.FloatTensor] = None