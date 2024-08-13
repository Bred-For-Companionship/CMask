from torch import nn
import torch
from models.vit import VisionTransformer, interpolate_pos_embed
from functools import partial
import torch.nn.functional as F
from models.rl.compute_reward import compute_reward

class CMaskAgent(nn.Module):
    def __init__(self, n_shared_layers=6, config=None, tokenizer=None, text_encoder=None):
        super().__init__()

        self.actions = []
        self.observations = []
        self.rewards = []

        self.mask_ratio = 0.3
        self.vision_proj = nn.Linear(config['vision_width'], config['embed_dim'])

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

        for param1, param2 in zip(main_model.vision_proj.parameters(), self.vision_proj.parameters()):
                param2.data.copy_(param1.data)

        #init final layers from last 3 layers of shared
        self.final_encoder = self.shared_text_encoder.bert.encoder.layer[-3:]
        for layer in self.final_encoder:
            self.shared_text_encoder.bert.encoder.layer.append(layer)

        return "params shared"

    def forward(self, image, text):
        #find lengths for each element by removing pad token first to calculate number of tokens to mask
        PAD_TOKEN_INDEX = self.shared_text_encoder.tokenizer.encode("[PAD]", tokenize=True)
        batch_lengths = (text != PAD_TOKEN_INDEX).sum(dim=1)

        mask_n = (0.3 * batch_lengths).long() #tensor representing number to mask for each row


        image_embeds = self.shared_vit(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.shared_text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha,
                                       output_attentions=True
                                       )

        probabilities = F.softmax(output[0], dim=1)

        for i in range(probabilities.size(0)):
            seq_prob = probabilities[i, :batch_lengths[i]]  # Shape: [sequence_length, num_actions]
            top_n = torch.topk(seq_prob, n[i], dim=0, largest=True, sorted=True).indices
            top_n_indices.append(top_n)

        mask_idx = torch.stack(top_n_indices)
        return {"image": image, "text": text, "mask_idx": mask_idx}


