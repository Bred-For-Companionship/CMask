from dataclasses import dataclass
from transformers.modeling_outputs import MaskedLMOutput
import torch
from typing import Tuple

@dataclass
class MaskedLMOutputCrossAttn(MaskedLMOutput):
    attentions: Tuple[torch.FloatTensor] = None