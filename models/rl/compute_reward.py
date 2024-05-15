import torch

def normalize_top_values(tensor):
    top_values, _ = torch.topk(tensor, k=8, dim=2)
    top_avg = torch.mean(top_values, dim=2)
    max_value = torch.max(top_avg)
    normalized_values = top_avg / max_value
    return normalized_values

def get_value_transformed_vectors(value_matrix, image_embeds):
    value_matrix = value_matrix.T.unsqueeze(0)
    result = value_matrix@image_embeds.unsqueeze(-1)
    return result.squeeze(-1)

def apply_value_transformed_vectors_to_attn_weights(attention_weights, value_transformed_vectors):
    return torch.norm(value_transformed_vectors * attention_weights.unsqueeze(1), p=2, dim=1)

def compute_reward(attention_weights, value_matrices, image_embeds):
    attention_weights = list(attention_weights)

    for layer_idx in range(len(attention_weights)):
        value_matrix = value_matrices[layer_idx]
        attn_matrix = attention_weights[layer_idx]

        value_transformed_vectors =  get_value_transformed_vectors(value_matrix, image_embeds)
        value_transformed_attn_norms - apply_value_transformed_vectors_to_attn_weights(attn_matrix,
                                                                                       value_transformed_vectors)

        attention_weights[layer_idx] = value_transformed_attn_norms

    attention_weights = tuple(attention_weights)

    value_weighted_attns = []

    for layer_attn_weights in attention_weights:
        value_weighted_attn = normalize_top_values(layer_attn_weights)
        value_weighted_attns.append(value_weighted_attn.unsqueeze(0))

    value_weighted_attns = torch.cat(value_weighted_attns, dim=1)
    reward_per_token = torch.mean(value_weighted_attns, dim=2)
    reward = torch.mean(reward_per_token, dim=1)

    return reward
