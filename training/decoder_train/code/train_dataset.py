import json
import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,DistributedSampler


class PromptSelector:
    def __init__(self, alpha=0.5, top_k=5, is_training=True, scale_encoding_type="ratio"):
        """"""
        self.alpha = alpha
        self.top_k = top_k if is_training else 1                
        self.is_training = is_training
        self.scale_encoding_type = scale_encoding_type

    def calculate_continuous_scale(self, selected_label, labels):
        """"""
                   
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
        """"""
                 
        unique_labels = torch.unique(labels)
        if len(unique_labels) == 0:
            raise ValueError("No valid labels found in point cloud!")

        if self.is_training:
            # Paper: randomly sample one target part per object.
            idx = torch.randint(0, unique_labels.numel(), (1,)).item()
            selected_label = unique_labels[idx].item()
        else:
            selected_label = unique_labels[0].item()

        continuous_scale, point_ratio = self.calculate_continuous_scale(selected_label, labels)

                  
        mask = (labels == selected_label)
        selected_indices = torch.where(mask)[0]
        selected_coords = coords[selected_indices]          

                   
        center = selected_coords.mean(dim=0)        
        center_distances = torch.norm(selected_coords - center, dim=1)               

                  
        other_mask = (labels != selected_label)
        other_indices = torch.where(other_mask)[0]
        other_coords = coords[other_indices]          

                            
        if len(other_indices) > 0:
                         
            pairwise_distances = torch.cdist(selected_coords, other_coords)          
            min_distances = pairwise_distances.min(dim=1)[0]                  
        else:
                                
            min_distances = torch.ones_like(center_distances) * 1e6

                       
        center_distances = (center_distances - center_distances.min()) / (
                    center_distances.max() - center_distances.min() + 1e-6)
        min_distances = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min() + 1e-6)

                
                                                      
                                     
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

class PointCloudDataset(Dataset):
    def __init__(self, metadata, feat_base_path=None):
        """"""
        if isinstance(metadata, str):
            with open(metadata, 'r') as f:
                self.metadata = json.load(f)
        else:
                             
            self.metadata = metadata

                               
        self.file_paths = []
        self.feat_paths = []
        
        if feat_base_path is None:
            raise ValueError(
                "feat_base_path is required. Set dataset.feat_base_path in the config "
                "(directory containing part_feat_<id>_0.npy)."
            )

        for item in self.metadata:
            self.file_paths.append(item['data_path'])
                        
            base_name = os.path.basename(item['data_path']).replace('.npy', '')
                        
            feat_path = os.path.join(feat_base_path, f'part_feat_{base_name}_0.npy')
            self.feat_paths.append(feat_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
                       
        return self.file_paths[idx], self.feat_paths[idx]


class CollateNpy:
    def __init__(self, num_points=10000, is_training=True):
        self.num_points = num_points
        self.is_training = is_training

    def __call__(self, batch_paths):
        return load_npy_to_dict(
            batch_paths, is_training=self.is_training, num_points=self.num_points
        )


def collate_fn(batch_paths):
    return load_npy_to_dict(batch_paths, is_training=True, num_points=10000)


def load_npy_to_dict(file_paths, is_training=True, num_points=10000):
    input_dict = {
        "coord": [],
        "feat": [],
        "label": [],
        "original_label": [],
    }
    point_counts = []

    for file_path, feat_path in file_paths:
        try:
            loaded_data = np.load(file_path, allow_pickle=True)
            if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == np.dtype('O'):
                loaded_data = loaded_data.item()

            feat_data = np.load(feat_path, allow_pickle=True)
            if isinstance(feat_data, np.ndarray) and feat_data.dtype == np.dtype('O'):
                feat_data = feat_data.item()
                if isinstance(feat_data, dict):
                    feat_data = feat_data['feat']

            if not isinstance(loaded_data, dict) or "coord" not in loaded_data or "label" not in loaded_data:
                raise ValueError(f"Missing required fields in {file_path}")

            coord_tensor = torch.tensor(loaded_data["coord"], dtype=torch.float32, requires_grad=False)
            feat_tensor = torch.tensor(feat_data, dtype=torch.float32, requires_grad=False)
            label_tensor = torch.tensor(loaded_data["label"], dtype=torch.long, requires_grad=False)

            if label_tensor.ndim == 1:
                pass
            elif label_tensor.ndim == 2 and label_tensor.size(1) == 1:
                label_tensor = label_tensor.squeeze(1)
            else:
                raise ValueError(f"Unsupported label shape {tuple(label_tensor.shape)} in {file_path}. Expected (N,) or (N,1).")

            n = coord_tensor.shape[0]
            if feat_tensor.shape[0] != n:
                raise ValueError(
                    f"Feature/coord length mismatch in {file_path}: coord={n}, feat={feat_tensor.shape[0]}"
                )
            if num_points is not None and n != num_points:
                replace = n < num_points
                indices = torch.from_numpy(
                    np.random.choice(n, num_points, replace=replace)
                ).long()
                coord_tensor = coord_tensor[indices]
                feat_tensor = feat_tensor[indices]
                label_tensor = label_tensor[indices]

            coord_tensor = (coord_tensor - coord_tensor.mean(dim=0, keepdim=True)) / (
                    coord_tensor.std(dim=0, keepdim=True) + 1e-6)

            input_dict["coord"].append(coord_tensor)
            input_dict["feat"].append(feat_tensor)
            input_dict["original_label"].append(label_tensor)
            point_counts.append(coord_tensor.shape[0])

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue

            
    if len(input_dict["coord"]) == 0:
        return None

    input_dict["coord"] = torch.cat(input_dict["coord"], dim=0)
    input_dict["feat"] = torch.cat(input_dict["feat"], dim=0)
                                                                            
    input_dict["original_label"] = torch.cat(input_dict["original_label"], dim=0)            

                 
    input_dict["batch"] = torch.cat(
        [torch.full((count,), i, dtype=torch.int32) for i, count in enumerate(point_counts)]
    )
    input_dict["grid_size"] = 0.01

                                              
                               
    prompt_selector = PromptSelector(
        alpha=0.5, 
        top_k=5 if is_training else 1,                    
        is_training=is_training,
        scale_encoding_type="ratio"            
    )
    batch_size = len(point_counts)
    prompts = []
    prompt_indices = []
    binary_labels = []
    continuous_scales = []         
    point_ratios = []
    selected_labels = []

    for i in range(batch_size):
        start_idx = sum(point_counts[:i])
        end_idx = start_idx + point_counts[i]
        coords = input_dict["coord"][start_idx:end_idx]
        original_labels = input_dict["original_label"][start_idx:end_idx]            

                                    
        (prompt_coord, prompt_idx, binary_label, 
         continuous_scale, point_ratio, selected_label) = prompt_selector.select_prompt(coords, original_labels)
        prompts.append(prompt_coord)
        prompt_indices.append(start_idx + prompt_idx)
        binary_labels.append(binary_label)
        continuous_scales.append(continuous_scale.item() if hasattr(continuous_scale, 'item') else continuous_scale)               
        point_ratios.append(point_ratio)
        selected_labels.append(selected_label)

          
    input_dict["prompt"] = torch.cat(prompts, dim=0)
    input_dict["prompt_indices"] = torch.tensor(prompt_indices, dtype=torch.long)
    input_dict["continuous_scales"] = torch.tensor(continuous_scales, dtype=torch.float32)         
    input_dict["point_ratios"] = torch.tensor(point_ratios, dtype=torch.float32)
    input_dict["selected_labels"] = torch.tensor(selected_labels, dtype=torch.long)
    
                                       
    stacked_labels = torch.stack(binary_labels, dim=0)
    if stacked_labels.ndim == 3:
                                                 
        input_dict["label"] = stacked_labels.squeeze(-1)
    else:
        input_dict["label"] = stacked_labels

    return input_dict


def make(config, rank, world_size):
    train_split_cfg = config.dataset.train_split

                                                         
    if isinstance(train_split_cfg, (str, bytes)):
        split_paths = [train_split_cfg]
    else:
        try:
            split_paths = list(train_split_cfg)
        except TypeError:
                             
            split_paths = [str(train_split_cfg)]

    merged_metadata = []
    per_file_counts = []
    for path in split_paths:
        try:
            path_str = str(path)
            with open(path_str, 'r') as f:
                md = json.load(f)
                count = len(md)
                per_file_counts.append((path_str, count))
                merged_metadata.extend(md)
        except Exception as e:
            if rank == 0:
                logging.error(f"Failed to load train split {path}: {e}")

                    
    if rank == 0:
        for p, c in per_file_counts:
            logging.info(f"Train split: {p} -> {c} samples")
        logging.info(f"Total train samples: {len(merged_metadata)}")

    feat_base_path = config.dataset.get("feat_base_path", None)
    dataset = PointCloudDataset(merged_metadata, feat_base_path=feat_base_path)
    batch_size = config.dataset.train_batch_size
    num_points = int(config.dataset.get("num_points", 10000))

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=CollateNpy(num_points=num_points, is_training=True),
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader