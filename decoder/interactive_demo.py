#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import argparse
import viser
import trimesh
from sklearn.decomposition import PCA
import time
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

encoder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'encoder')
sys.path.insert(0, encoder_path)

from param import parse_args
from utils.misc import load_config
from models import make_PointFeatureEnhancer, make_decoder, make_seg_head
from demo_dataloader import PromptSelector

try:
    from partfield.config import default_argument_parser as encoder_parse_args, setup as encoder_setup
    from partfield.model_trainer_pvcnn_only_demo import Model as EncoderModel
    from partfield.model.PVCNN.encoder_pc import sample_triplane_feat
    ENCODER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cannot import encoder module: {e}")
    ENCODER_AVAILABLE = False

POINT_COLOR = np.array([173, 216, 230])
PROMPT_COLOR = np.array([0, 255, 0])
MASK_COLOR = np.array([255, 182, 193])

POINT_SIZE = 0.005
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))


class S2AM3DInference:
    
    def __init__(self, config, checkpoint_path, device='cuda:0'):
        self.config = config
        self.device = device
        
        self.PointFeatureEnhancer = make_PointFeatureEnhancer(config).to(device)
        self.decoder = make_decoder(config).to(device)
        self.seg_head = make_seg_head(config).to(device)
        
        self.load_checkpoint(checkpoint_path)
        
        self.PointFeatureEnhancer.eval()
        self.decoder.eval()
        self.seg_head.eval()
        
        self.prompt_selector = PromptSelector(
            alpha=0.5,
            top_k=1,
            is_training=False,
            scale_encoding_type="ratio"
        )
        
        self.enhancefeat_dim = config.enhancer.enhancefeat_dim
        print(f"Model loaded on {device}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        def remove_module_prefix(state_dict):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            return new_state_dict
        
        enhancer_state_dict = remove_module_prefix(checkpoint['point_feature_enhancer_state_dict'])
        decoder_state_dict = remove_module_prefix(checkpoint['decoder_state_dict'])
        seg_head_state_dict = remove_module_prefix(checkpoint['seg_head_state_dict'])
        
        self.PointFeatureEnhancer.load_state_dict(enhancer_state_dict, strict=False)
        self.decoder.load_state_dict(decoder_state_dict)
        self.seg_head.load_state_dict(seg_head_state_dict)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    @torch.no_grad()
    def predict_mask(self, point_feat, point_coords, point_color, prompt_idx, threshold=0.5, continuous_scale=None):
        batch_size = 1
        actual_num_points = point_feat.shape[0]
        
        point_feat = point_feat.view(batch_size, actual_num_points, -1).to(self.device)
        point_coords = point_coords.view(batch_size, actual_num_points, 3).to(self.device)
        
        if point_color is not None:
            point_color = point_color.view(batch_size, actual_num_points, 3).to(self.device)
        else:
            point_color = torch.ones(batch_size, actual_num_points, 3).to(self.device)
        
        continuous_scales = None
        if continuous_scale is not None and self.config.get('use_continuous_scale', True):
            continuous_scales = torch.tensor([continuous_scale], device=self.device).float()
            if continuous_scales.dim() > 1:
                continuous_scales = continuous_scales.view(-1)
        
        enhance_feat = self.PointFeatureEnhancer(point_feat, point_coords, point_color, continuous_scales)
        
        enhance_feat = enhance_feat.view(batch_size * actual_num_points, self.enhancefeat_dim)
        prompt_feat = enhance_feat[prompt_idx:prompt_idx+1].view(batch_size, 1, self.enhancefeat_dim)
        
        enhance_feat = enhance_feat.view(batch_size, actual_num_points, self.enhancefeat_dim)
        decoder_output = self.decoder(enhance_feat, prompt_feat)
        
        seg_pred = self.seg_head(decoder_output)
        seg_pred = seg_pred[0].cpu().numpy()  # (N,)
        
        mask = seg_pred > threshold
        confidence = float(np.max(seg_pred))
        
        return mask, confidence, seg_pred


def normalize_pc(pc):
    max_, min_ = np.max(pc, axis=0), np.min(pc, axis=0)
    center = (max_ + min_) / 2
    scale = (max_ - min_) / 2
    scale = np.max(np.abs(scale)) + 1e-10
    pc = (pc - center) / scale
    return pc


class FeatureExtractor:
    
    def __init__(self, encoder_config_path, encoder_ckpt_path, device='cuda:0'):
        if not ENCODER_AVAILABLE:
            raise RuntimeError("Encoder module not available")
        
        self.device = device
        
        encoder_args = encoder_parse_args()
        encoder_args.config_file = encoder_config_path
        encoder_args.opts = []
        self.encoder_cfg = encoder_setup(encoder_args, freeze=False)
        
        self.encoder_model = EncoderModel(self.encoder_cfg)
        
        checkpoint = torch.load(encoder_ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
        self.encoder_model.load_state_dict(state_dict, strict=True)
        self.encoder_model.eval()
        self.encoder_model.to(device)
        
        print(f"Encoder loaded: {encoder_ckpt_path}")
    
    @torch.no_grad()
    def extract_features(self, coords, colors=None):
        coords_norm = normalize_pc(coords)
        
        pc_tensor = torch.from_numpy(coords_norm).float().unsqueeze(0).to(self.device)  # (1, N, 3)
        
        if colors is None:
            colors = coords_norm
        color_tensor = torch.from_numpy(colors).float().unsqueeze(0).to(self.device)  # (1, N, 3)
        
        pc_feat = self.encoder_model.pvcnn(pc_tensor, color_tensor)
        
        # 2. Triplane Transformer
        planes = self.encoder_model.triplane_transformer(pc_feat)  # (B, 3, C', H, W)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
        
        point_feat = sample_triplane_feat(part_planes, pc_tensor)  # (1, N, 448)
        point_feat = point_feat.squeeze(0).cpu().numpy()  # (N, 448)
        
        return point_feat


def load_pointcloud_from_npy(npy_path, feature_extractor=None):
    data = np.load(npy_path, allow_pickle=True).item()
    
    coords = data.get("coord", None)
    if coords is None:
        raise ValueError(f"Missing 'coord' in {npy_path}")
    
    colors = data.get("color", None)
    if colors is None:
        colors = np.ones((len(coords), 3), dtype=np.float32)
    
    if feature_extractor is not None:
        try:
            feats = feature_extractor.extract_features(coords, colors)
        except Exception as e:
            raise ValueError(f"Feature extraction failed: {e}")
    else:
        feats = data.get("feat", None)
        if feats is None:
            raise ValueError("Features not found. Provide --encoder_config and --encoder_ckpt or include 'feat' in .npy file")
    
    return coords, feats, colors


def load_pointcloud_from_mesh(mesh_path, num_points=10000, feature_extractor=None):
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    colors = np.ones((len(points), 3), dtype=np.float32)
    
    if feature_extractor is not None:
        feats = feature_extractor.extract_features(points, colors)
    else:
        raise ValueError("Feature extractor required for mesh loading")
    
    return points, feats, colors


def mask2color(mask, base_color=POINT_COLOR, mask_color=MASK_COLOR):
    point_num = mask.shape[0]
    colors = np.tile(base_color, (point_num, 1))
    colors[mask] = mask_color
    return colors


def main(args):
    cli_args, extras = parse_args([args.config])
    config = load_config(args.config, cli_args=vars(cli_args), extra_args=extras)
    
    model = S2AM3DInference(config, args.ckpt_path, device=args.device)
    
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    
    points = [None]
    points_handle = [None]
    feats = [None]
    colors = [None]
    colors_pca = [None]
    show_colors = [None]
    point_prompt = [None]
    mask_res = [None]
    confidence_res = [None]
    load_error_msg = [None]
    data_files = [[]]  # list of dicts: {'id': id_str, 'path': full_path}
    data_id_to_path = [{}]
    
    def remove_point_prompt():
        if point_prompt[0] is not None:
            server.scene.remove_by_name("/prompt_sphere")
            point_prompt[0] = None
    
    def clear_state():
        mask_res[0] = None
        confidence_res[0] = None
        remove_point_prompt()
        if points_handle[0] is not None:
            points_handle[0].colors = show_colors[0]
    
    feature_extractor = None
    if args.encoder_config and args.encoder_ckpt:
        try:
            feature_extractor = FeatureExtractor(
                args.encoder_config,
                args.encoder_ckpt,
                device=args.device
            )
        except Exception as e:
            print(f"Warning: Cannot load encoder: {e}")

    def refresh_data_files():
        if args.data_dir and os.path.isdir(args.data_dir):
            entries = []
            for f in sorted(
                os.listdir(args.data_dir)
            ):
                if not f.endswith('.npy'):
                    continue
                full_path = os.path.join(args.data_dir, f)
                file_id = os.path.splitext(f)[0]
                entries.append({"id": file_id, "path": full_path})
            data_files[0] = entries
            data_id_to_path[0] = {e["id"]: e["path"] for e in entries}
        else:
            data_files[0] = []
            data_id_to_path[0] = {}
    
    def load_pc(file_path=None):
        clear_state()
        load_error_msg[0] = None
        
        if file_path is None:
            refresh_data_files()
            if data_files[0]:
                file_path = data_files[0][0]["path"]
            else:
                if args.data_dir:
                    if not os.path.isdir(args.data_dir):
                        error_msg = f"Error: Data directory not found: {args.data_dir}"
                    else:
                        error_msg = f"Error: No .npy files found in {args.data_dir}"
                else:
                    error_msg = "Error: No data path specified"
                load_error_msg[0] = error_msg
                return
        
        try:
            if file_path.endswith('.npy'):
                _coords, _feats, _colors = load_pointcloud_from_npy(
                    file_path, 
                    feature_extractor=feature_extractor
                )
            elif file_path.endswith(('.obj', '.glb', '.ply')):
                if feature_extractor is None:
                    error_msg = "Error: Encoder config and checkpoint required for mesh loading"
                    load_error_msg[0] = error_msg
                    return
                _coords, _feats, _colors = load_pointcloud_from_mesh(
                    file_path, 
                    num_points=args.point_num,
                    feature_extractor=feature_extractor
                )
            else:
                error_msg = f"Error: Unsupported file type: {file_path}"
                load_error_msg[0] = error_msg
                return
        except Exception as e:
            error_msg = f"Failed to load point cloud: {e}"
            load_error_msg[0] = error_msg
            return
        
        _coords = normalize_pc(_coords)
        
        _feats_tensor = torch.from_numpy(_feats).float()
        _coords_tensor = torch.from_numpy(_coords).float()
        _colors_tensor = torch.from_numpy(_colors).float()
        
        if _feats.shape[1] >= 3:
            try:
                feat_norm = _feats / (np.linalg.norm(_feats, axis=-1, keepdims=True) + 1e-8)
                pca = PCA(n_components=3)
                feat_reduced = pca.fit_transform(feat_norm)
                feat_reduced = (feat_reduced - feat_reduced.min()) / (feat_reduced.max() - feat_reduced.min() + 1e-8)
                _colors_pca = (feat_reduced * 255).astype(np.uint8)
            except:
                _colors_pca = np.ones((len(_coords), 3), dtype=np.uint8) * 128
        else:
            _colors_pca = np.ones((len(_coords), 3), dtype=np.uint8) * 128
        
        _show_colors = np.tile(POINT_COLOR, (len(_coords), 1))
        
        _points_handle = server.scene.add_point_cloud(
            name="/point_cloud",
            points=_coords,
            colors=_show_colors,
            point_size=POINT_SIZE,
        )
        
        points[0] = _coords
        points_handle[0] = _points_handle
        feats[0] = _feats_tensor
        colors[0] = _colors_tensor
        colors_pca[0] = _colors_pca
        show_colors[0] = _show_colors
        
        print(f"Point cloud loaded: {len(_coords)} points")
    
    initial_load_done = [False]
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        title_markdown = client.gui.add_markdown(
            """
            # 🎯 S2AM3D Interactive Segmentation
            
            **Interactive 3D Point Cloud Part Segmentation**
            """
        )
        
        data_section = client.gui.add_folder("📁 Data Loading")
        with data_section:
            data_path_handle = client.gui.add_text(
                "Object ID / Path", 
                initial_value="",
                hint="Enter object ID (from dropdown) or full path"
            )

            data_dropdown_handle = client.gui.add_dropdown(
                "Objects in data_dir",
                options=("Scanning...",),
            )

            refresh_button_handle = client.gui.add_button(
                "🔃 Refresh data_dir list", icon=viser.Icon.REFRESH
            )
            
            load_button_handle = client.gui.add_button(
                "🔄 Load Point Cloud", icon=viser.Icon.REFRESH
            )
        
        interaction_section = client.gui.add_folder("✏️ Interaction")
        with interaction_section:
            click_button_handle = client.gui.add_button(
                "📍 Select Point Prompt", icon=viser.Icon.POINTER
            )
            
            clear_button_handle = client.gui.add_button(
                "🗑️ Clear Selection", icon=viser.Icon.X
            )
        
        params_section = client.gui.add_folder("⚙️ Segmentation Parameters")
        with params_section:
            use_scale_checkbox = client.gui.add_checkbox(
                "📐 Use Scale",
                initial_value=True
            )

            scale_slider = client.gui.add_slider(
                "🎚️ Segmentation Scale",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.3,
                hint="Adjust the scale of segmentation (0.0 = small parts, 1.0 = large parts)"
            )
            scale_slider.disabled = not use_scale_checkbox.value
            
            scale_dec_button = client.gui.add_button("➖ Fine -")
            scale_inc_button = client.gui.add_button("➕ Fine +")
            
            scale_ultra_dec_button = client.gui.add_button("➖➖ Ultra Fine -")
            scale_ultra_inc_button = client.gui.add_button("➕➕ Ultra Fine +")
            
            scale_dec_button.disabled = not use_scale_checkbox.value
            scale_inc_button.disabled = not use_scale_checkbox.value
            scale_ultra_dec_button.disabled = not use_scale_checkbox.value
            scale_ultra_inc_button.disabled = not use_scale_checkbox.value
            THRESHOLD_FIXED = 0.7
        
        display_section = client.gui.add_folder("👁️ Display Options")
        with display_section:
            show_feature_checkbox = client.gui.add_checkbox(
                "🌈 Show Feature Colors", initial_value=False,
                hint="Display point cloud colored by extracted features"
            )
            
            point_size_slider = client.gui.add_slider(
                "🔍 Point Size",
                min=0.005,
                max=0.01,
                step=0.0005,
                initial_value=POINT_SIZE,
                hint="Adjust the size of points"
            )
        
        status_section = client.gui.add_folder("📊 Status & Results")
        with status_section:
            info_markdown = client.gui.add_markdown(
                f"""
                ### 📈 Current Status
                
                **Status**: ⏳ Waiting for point cloud...
                
                **Confidence**: `-`
                
                **Scale**: `{'enabled (0.300)' if use_scale_checkbox.value else 'disabled'}`
                
                ---
                
                ### 🔧 System Info
                
                **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded (will read from file)'}
                """
            )

        def update_data_dropdown():
            refresh_data_files()
            if data_files[0]:
                options = tuple(e["id"] for e in data_files[0])
                data_dropdown_handle.options = options
                if data_dropdown_handle.value not in options:
                    data_dropdown_handle.value = data_files[0][0]["id"]
                if not data_path_handle.value.strip():
                    data_path_handle.value = data_dropdown_handle.value
            else:
                placeholder = "No .npy found in data_dir" if args.data_dir else "Set data_dir or data_path"
                data_dropdown_handle.options = (placeholder,)
                data_dropdown_handle.value = placeholder

        update_data_dropdown()

        @data_dropdown_handle.on_update
        def _(_):
            if data_files[0] and data_dropdown_handle.value in data_id_to_path[0]:
                data_path_handle.value = data_dropdown_handle.value

        @refresh_button_handle.on_click
        def _(_):
            update_data_dropdown()
        
        def get_scale_status_text():
            if not use_scale_checkbox.value:
                return "disabled"
            return f"enabled ({scale_slider.value:.3f})"

        def show_mask():
            if points_handle[0] is None:
                return
            
            if not show_feature_checkbox.value:
                if mask_res[0] is not None:
                    mask_colors = mask2color(mask_res[0])
                    points_handle[0].colors = mask_colors
                    mask_points = np.sum(mask_res[0])
                    total_points = len(mask_res[0])
                    mask_ratio = mask_points / total_points * 100
                    info_markdown.content = f"""
                    ### 📈 Current Status
                    
                    **Status**: ✅ **Segmented**
                    
                    **Confidence**: `{confidence_res[0]:.3f}`
                    
                    **Mask Points**: `{mask_points:,}` / `{total_points:,}` (`{mask_ratio:.1f}%`)

                    **Scale**: `{get_scale_status_text()}`
                    
                    ---
                    
                    ### 🔧 System Info
                    
                    **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
                    """
                else:
                    points_handle[0].colors = show_colors[0]
                    info_markdown.content = f"""
                    ### 📈 Current Status
                    
                    **Status**: ⏳ Waiting for point prompt...
                    
                    **Confidence**: `-`

                    **Scale**: `{get_scale_status_text()}`
                    
                    ---
                    
                    ### 🔧 System Info
                    
                    **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
                    """
            else:
                if colors_pca[0] is not None:
                    points_handle[0].colors = colors_pca[0]
                else:
                    points_handle[0].colors = show_colors[0]
        
        def add_point_prompt():
            if point_prompt[0] is not None:
                server.scene.add_icosphere(
                    name="/prompt_sphere",
                    radius=0.01,
                    color=PROMPT_COLOR,
                    position=point_prompt[0],
                )
        
        @load_button_handle.on_click
        def _(_):
            user_value = data_path_handle.value.strip()
            file_path = None

            # Prioritize id selection
            if user_value in data_id_to_path[0]:
                file_path = data_id_to_path[0][user_value]

            # If user typed a full/relative path
            if file_path is None and user_value and os.path.exists(user_value):
                file_path = user_value

            # Fallback to dropdown selection
            if file_path is None and data_files[0]:
                selected_id = data_dropdown_handle.value
                if selected_id in data_id_to_path[0]:
                    file_path = data_id_to_path[0][selected_id]

            if file_path:
                load_pc(file_path)
                if load_error_msg[0] is not None:
                    info_markdown.content = f"""
                    ### ❌ Error
                    
                    **Message**: {load_error_msg[0]}
                    
                    ---
                    
                    ### 🔧 System Info
                    
                    **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
                    """
                else:
                    show_mask()
            else:
                info_markdown.content = f"""
                ### ⚠️ Warning
                
                **Message**: No file selected. Choose an ID from data_dir dropdown or type a valid path/ID.
                
                ---
                
                ### 🔧 System Info
                
                **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
                """
        
        @click_button_handle.on_click
        def _(_):
            if points[0] is None:
                info_markdown.content = """
                ### ⚠️ Warning
                
                **Message**: Please load a point cloud first!
                
                ---
                
                ### 🔧 System Info
                
                **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
                """
                return
            
            click_button_handle.disabled = True
            info_markdown.content = """
            ### 📍 Interactive Mode
            
            **Status**: Click on the point cloud to select a prompt point...
            
            **Instructions**: 
            - Click anywhere on the point cloud
            - The system will segment based on your selection
            
            ---
            
            ### 🔧 System Info
            
            **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
            """
            
            @client.scene.on_pointer_event(event_type="click")
            def _(event: viser.ScenePointerEvent) -> None:
                o = np.array(event.ray_origin)
                d = np.array(event.ray_direction)
                
                A = points[0] - o
                B = np.expand_dims(d, axis=0)
                AB = np.sum(A * B, axis=-1)
                B_square = np.sum(B ** 2, axis=-1)
                t = AB / B_square
                intersect_points = o + t.reshape(-1, 1) * d
                distv = np.sum((intersect_points - points[0]) ** 2, axis=-1) ** 0.5
                disth = t * np.sqrt(B_square)
                
                mask = (distv < POINT_SIZE)
                if np.sum(mask) == 0:
                    mask = (distv < POINT_SIZE * 5)
                    if np.sum(mask) == 0:
                        client.scene.remove_pointer_callback()
                        click_button_handle.disabled = False
                        return
                
                select_points = points[0][mask]
                disth = disth[mask]
                min_disth_idx = np.argmin(disth)
                select_point = select_points[min_disth_idx]
                select_idx = np.where(np.all(points[0] == select_point, axis=1))[0][0]
                
                print(f"Selected prompt point: {select_point}, index: {select_idx}")
                point_prompt[0] = select_point
                add_point_prompt()
                
                info_markdown.content = """
                ### ⚙️ Processing...
                
                **Status**: Computing segmentation...
                
                **Please wait...**
                
                ---
                
                ### 🔧 System Info
                
                **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
                """
                
                continuous_scale = None
                if model.config.get('use_continuous_scale', True):
                    continuous_scale = scale_slider.value
                scale_log = f"{continuous_scale:.3f}" if continuous_scale is not None else "None"
                print(f"Predicting... (scale={scale_log}, threshold={THRESHOLD_FIXED:.3f})")
                mask, confidence, seg_pred = model.predict_mask(
                    feats[0],
                    torch.from_numpy(points[0]).float(),
                    colors[0],
                    select_idx,
                    threshold=THRESHOLD_FIXED,
                    continuous_scale=continuous_scale
                )
                
                mask_res[0] = mask
                confidence_res[0] = confidence
                
                print(f"Prediction complete: mask points={np.sum(mask)}, confidence={confidence:.3f}")
                show_mask()
                
                client.scene.remove_pointer_callback()
            
            @client.scene.on_pointer_callback_removed
            def _():
                click_button_handle.disabled = False
        
        @clear_button_handle.on_click
        def _(_):
            clear_state()
            show_mask()
        
        def recompute_with_current_scale():
            if mask_res[0] is None or point_prompt[0] is None:
                show_mask()
                return
            info_markdown.content = """
            ### ⚙️ Updating...
            
            **Status**: Recomputing with current scale option...
            
            **Please wait...**
            
            ---
            
            ### 🔧 System Info
            
            **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
            """
            select_idx = np.where(np.all(points[0] == point_prompt[0], axis=1))[0][0]
            if model.config.get('use_continuous_scale', True) and use_scale_checkbox.value:
                continuous_scale = scale_slider.value
                scale_log = f"{continuous_scale:.3f}"
            else:
                continuous_scale = None
                scale_log = "None"
            print(f"Updating prediction... (scale={scale_log}, threshold={THRESHOLD_FIXED:.3f})")
            mask, confidence, _ = model.predict_mask(
                feats[0],
                torch.from_numpy(points[0]).float(),
                colors[0],
                select_idx,
                threshold=THRESHOLD_FIXED,
                continuous_scale=continuous_scale
            )
            mask_res[0] = mask
            confidence_res[0] = confidence
            show_mask()

        def adjust_scale(delta):
            if not use_scale_checkbox.value:
                return
            new_val = max(0.0, min(1.0, scale_slider.value + delta))
            if abs(new_val - scale_slider.value) < 1e-9:
                return
            scale_slider.value = new_val
            recompute_with_current_scale()

        @scale_dec_button.on_click
        def _(_):
            adjust_scale(-0.01)

        @scale_inc_button.on_click
        def _(_):
            adjust_scale(0.01)

        @scale_ultra_dec_button.on_click
        def _(_):
            adjust_scale(-0.001)

        @scale_ultra_inc_button.on_click
        def _(_):
            adjust_scale(0.001)

        @scale_slider.on_update
        def _(_):
            if not use_scale_checkbox.value:
                return
            recompute_with_current_scale()

        @show_feature_checkbox.on_update
        def _(_):
            show_mask()
        
        @point_size_slider.on_update
        def _(_):
            global POINT_SIZE
            if points_handle[0] is not None:
                points_handle[0].point_size = point_size_slider.value
                POINT_SIZE = point_size_slider.value
        
        @use_scale_checkbox.on_update
        def _(_):
            scale_slider.disabled = not use_scale_checkbox.value
            scale_dec_button.disabled = not use_scale_checkbox.value
            scale_inc_button.disabled = not use_scale_checkbox.value
            scale_ultra_dec_button.disabled = not use_scale_checkbox.value
            scale_ultra_inc_button.disabled = not use_scale_checkbox.value
            recompute_with_current_scale()
        
        if not initial_load_done[0]:
            if args.data_path:
                load_pc(args.data_path)
            elif args.data_dir:
                load_pc()
            
            if load_error_msg[0] is not None:
                info_markdown.content = f"""
                ### ❌ Error
                
                **Message**: {load_error_msg[0]}
                
                ---
                
                ### 🔧 System Info
                
                **Encoder**: {'✅ Loaded' if feature_extractor is not None else '⚠️ Not loaded'}
                """
            else:
                show_mask()
            
            initial_load_done[0] = True
    
    print(f"\nServer started: http://{args.host}:{args.port}")
    
    while True:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2AM3D Interactive Point Cloud Segmentation Demo")
    parser.add_argument('--config', type=str, required=True, help='Decoder config path')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Decoder checkpoint path')
    parser.add_argument('--data_path', type=str, default=None, help='Point cloud data path (.npy file)')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Point cloud data directory (default: {DEFAULT_DATA_DIR})')
    
    parser.add_argument('--encoder_config', type=str, default=None,
                        help='Encoder config path')
    parser.add_argument('--encoder_ckpt', type=str, default=None,
                        help='Encoder checkpoint path')
    
    parser.add_argument('--host', default="0.0.0.0", help='Server host')
    parser.add_argument('--port', default=8080, type=int, help='Server port')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--point_num', default=10000, type=int, help='Number of points to sample from mesh')
    
    args = parser.parse_args()
    
    main(args)
