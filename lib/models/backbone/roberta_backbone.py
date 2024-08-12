
import torch
from torch import nn

from transformers import RobertaModel


class RobertaLanguageBackbone(nn.Module):
    """
    Roberta Language backbone, using the Hugging Face transformers library.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        text_encoder = RobertaModel.from_pretrained(config.text_encoder_type)
        
        self.model = text_encoder
        self.hidden_size = text_encoder.config.hidden_size # 768
        self.num_text_output_layers = config.num_text_output_layers # total 12 layers for base roberta
        
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor=None):
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        # convert attention mask to boolean, True for padding tokens.
        attention_mask = attention_mask.ne(1).bool()
        hidden_states = outputs.hidden_states[-self.num_text_output_layers:]
        
        return hidden_states, attention_mask
