import torch


class PromptSelector:
    """Lightweight prompt selector kept for demos."""

    def __init__(self, alpha=0.5, top_k=5, is_training=True, scale_encoding_type="ratio"):
        self.alpha = alpha
        self.top_k = top_k if is_training else 1  # deterministic at inference
        self.is_training = is_training
        self.scale_encoding_type = scale_encoding_type

    def calculate_continuous_scale(self, selected_label, labels):
        """Return scale and ratio based on label coverage."""
        mask = (labels == selected_label)
        selected_point_count = mask.sum().item()
        total_point_count = len(labels)

        point_ratio = selected_point_count / total_point_count

        if self.scale_encoding_type == "ratio":
            continuous_scale = point_ratio
        elif self.scale_encoding_type == "log_ratio":
            continuous_scale = torch.log(torch.tensor(point_ratio) + 1e-8)
        elif self.scale_encoding_type == "normalized_ratio":
            continuous_scale = 2 * torch.sigmoid(torch.tensor(point_ratio) * 10 - 5) - 1
        else:
            continuous_scale = point_ratio

        return continuous_scale, point_ratio

    def select_prompt(self, coords, labels):
        """Pick a prompt point and binary labels."""
        unique_labels = torch.unique(labels)
        if len(unique_labels) == 0:
            raise ValueError("No valid labels found in point cloud!")

        # deterministic label selection for demo
        selected_label = unique_labels[0].item()

        continuous_scale, point_ratio = self.calculate_continuous_scale(selected_label, labels)

        mask = (labels == selected_label)
        selected_indices = torch.where(mask)[0]
        selected_coords = coords[selected_indices]  # (M, 3)

        center = selected_coords.mean(dim=0)  # (3,)
        center_distances = torch.norm(selected_coords - center, dim=1)  # (M,)

        other_mask = (labels != selected_label)
        other_indices = torch.where(other_mask)[0]
        other_coords = coords[other_indices]  # (K, 3)

        if len(other_indices) > 0:
            pairwise_distances = torch.cdist(selected_coords, other_coords)  # (M, K)
            min_distances = pairwise_distances.min(dim=1)[0]  # (M,)
        else:
            min_distances = torch.ones_like(center_distances) * 1e6

        center_distances = (center_distances - center_distances.min()) / (
            center_distances.max() - center_distances.min() + 1e-6
        )
        min_distances = (min_distances - min_distances.min()) / (
            min_distances.max() - min_distances.min() + 1e-6
        )

        scores = self.alpha * (1 - center_distances) + (1 - self.alpha) * min_distances

        if self.is_training and self.top_k > 1:
            top_k_indices = torch.topk(scores, min(self.top_k, len(scores)))[1]
            prompt_local_idx = top_k_indices[0].item()
        else:
            prompt_local_idx = scores.argmax().item()

        prompt_idx = selected_indices[prompt_local_idx].item()
        prompt_coords = coords[prompt_idx].unsqueeze(0)

        binary_labels = (labels == selected_label).long()

        return prompt_coords, prompt_idx, binary_labels, continuous_scale, point_ratio, selected_label
