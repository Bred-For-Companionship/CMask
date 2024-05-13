import torch

def normalize_top_values(tensor):
    top_values, _ = torch.topk(tensor, k=8, dim=1)
    top_avg = torch.mean(top_values, dim=1)
    max_value = torch.max(top_avg)
    normalized_values = top_avg / max_value
    return normalized_values

def compute_reward(attention_weights, value_matrices, image_embeds):
    value_weighted_attns = []

    for layer_attn_weights in attention_weights:
        value_weighted_attn = normalize_top_values(layer_attn_weights)
        value_weighted_attns.append(value_weighted_attn.unsqueeze(0))

    value_weighted_attns = torch.cat(value_weighted_attns, dim=0)
    reward_per_token = torch.mean(value_weighted_attns, dim=1)
    reward = torch.mean(reward_per_token)

    return reward
