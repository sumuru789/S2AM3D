from yacs.config import CfgNode as CN

_C = CN()
_C.seed = 0
_C.output_dir = "results"
_C.result_name = "test_all"
_C.output_feature_dir = ""

_C.triplet_sampling = "random"
_C.load_original_mesh = False

_C.num_pos = 64
_C.num_neg_random = 256
_C.num_neg_hard_pc = 128
_C.num_neg_hard_emb = 128

_C.vertex_feature = False  # if true, sample feature on vertices; if false, sample feature on faces
_C.n_point_per_face = 2000
_C.n_sample_each = 10000
_C.preprocess_mesh = False

_C.regress_2d_feat = False

_C.is_pc = False

_C.cut_manifold = False
_C.remesh_demo = False
_C.correspondence_demo = False

_C.save_every_epoch = 10
_C.training_epochs = 30
_C.continue_training = False

_C.continue_ckpt = None
_C.epoch_selected = "epoch=50.ckpt"

_C.triplane_resolution = 128
_C.triplane_channels_low = 128
_C.triplane_channels_high = 512
_C.lr = 1e-3
_C.train = True
_C.test = False

_C.inference_save_pred_sdf_to_mesh=True
_C.inference_save_feat_pca=True
_C.name = "test"
_C.test_subset = False
_C.test_corres = False
_C.test_partobjaversetiny = False

_C.dataset = CN()
_C.dataset.type = "Demo_Dataset"
_C.dataset.data_path = "objaverse_data/"
_C.dataset.train_num_workers = 64
_C.dataset.val_num_workers = 32
_C.dataset.train_batch_size = 2
_C.dataset.val_batch_size = 2
_C.dataset.all_files = []  # only used for correspondence demo
_C.dataset.train_metadata_paths = []  # optional: list of metadata json paths for training; if empty, fallback to defaults

_C.voxel2triplane = CN()
_C.voxel2triplane.transformer_dim = 1024
_C.voxel2triplane.transformer_layers = 6
_C.voxel2triplane.transformer_heads = 8
_C.voxel2triplane.triplane_low_res = 32
_C.voxel2triplane.triplane_high_res = 256
_C.voxel2triplane.triplane_dim = 64
_C.voxel2triplane.normalize_vox_feat = False


_C.loss = CN()
_C.loss.triplet = 0.0
_C.loss.sdf = 1.0
_C.loss.feat = 10.0
_C.loss.l1 = 0.0

_C.use_pvcnn = False
_C.use_pvcnnonly = True

_C.pvcnn = CN()
_C.pvcnn.point_encoder_type = 'pvcnn'
_C.pvcnn.use_point_scatter = True
_C.pvcnn.z_triplane_channels = 64
_C.pvcnn.z_triplane_resolution = 256
_C.pvcnn.unet_cfg = CN()
_C.pvcnn.unet_cfg.depth = 3
_C.pvcnn.unet_cfg.enabled = True
_C.pvcnn.unet_cfg.rolled = True
_C.pvcnn.unet_cfg.use_3d_aware = True
_C.pvcnn.unet_cfg.start_hidden_channels = 32
_C.pvcnn.unet_cfg.use_initial_conv = False

_C.use_2d_feat = False
_C.inference_metrics_only = False

_C.freeze_backbone = False
_C.optimizer = CN()
_C.optimizer.lr = 1e-4
_C.optimizer.weight_decay = 1e-4

_C.sampling_strategy = 'triplet'
_C.num_points = 1000
_C.dataset_num_points = 10000
_C.contrast_num_points = 500

_C.batch_size = 1

_C.use_color = True

_C.source_loss_weight = CN()
_C.source_loss_weight.partnet = 1.0
_C.source_loss_weight.objaverse = 1.0
_C.source_loss_weight.unknown = 1.0
