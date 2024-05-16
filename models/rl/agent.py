from torch import nn
import torch
from models.vit import VisionTransformer, interpolate_pos_embed
from functools import partial

class CMaskAgent(nn.Module):
    def __init__(self, n_shared_layers=6, config=None, tokenizer=None, text_encoder=None):
        super().__init__()

        self.n_shared_layers = n_shared_layers

        if config is not None:
            self.shared_vit = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

            if text_encoder is not None:
                self.shared_text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
            else:
                self.shared_text_encoder = BertForMaskedLM.from_pretrained("bert-base-uncased", config=bert_config)

        else:
            self.shared_vit = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.shared_text_encoder = BertForMaskedLM.from_pretrained("bert-base-uncased")


    def share_params_first_n_layers(self, main_model):
        flag = self.n_shared_layers
        for param1, param2 in zip(main_model.text_encoder.parameters(), self.shared_text_encoder.parameters()):
            if flag != 0:
                param2.data.copy_(param1.data)
                flag -= 1

        flag = self.n_shared_layers
        for param1, param2 in zip(main_model.visual_encoder.parameters(), self.shared_vit.parameters()):
            if flag != 0:
                param2.data.copy_(param1.data)
                flag -= 1
        return "params shared"