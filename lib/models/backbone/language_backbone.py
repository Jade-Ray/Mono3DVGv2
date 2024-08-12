from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers.utils import ModelOutput


@dataclass
class LanguageBackboneOutput(ModelOutput):
    """
    Base class for Language Backbone's outputs, with potential hidden states and attention masks.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **masked**,
            - 0 for tokens that are **not masked**.
    """
    last_hidden_state: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class LanguageBackboneModel(nn.Module):
    """
    This module tonkenizes text strings to the language backbone.
    """

    def __init__(self, backbone, tokenizer):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer

    def forward(self, text, device, return_dict=None,):
        # send text through tokenizer to get input_ids and attention_mask.
        tokenized = self.tokenizer.batch_encode_plus(
            text, max_length=110, padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)
        hidden_states, attention_mask = self.backbone(**tokenized)
        
        if not return_dict:
            return hidden_states, attention_mask
        return LanguageBackboneOutput(
            last_hidden_state=hidden_states[-1], hidden_states=hidden_states, attention_mask=attention_mask,
        )
