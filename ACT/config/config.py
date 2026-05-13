import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR =  'E:\ACT-PickandPlace\data\pick_and_place_4cam_absjoint_noisy_goalqpos'

# checkpoint directory
CHECKPOINT_DIR = 'checkpoints/'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

# robot port names
ROBOT_PORTS = {
    'leader': '/dev/tty.usbmodem57380045221',
    'follower': '/dev/tty.usbmodem57380046991'
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 100,
    'state_dim': 11,            # 7 joints + 1 gripper width
    'action_dim': 8,
    'cam_width': 320,
    'cam_height': 240,
    'camera_names': ['top', 'front', 'side','wrist'],
    'num_episodes': 100,
    # leader_port / follower_port / camera_port not needed for sim
}

# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 50,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['top', 'front', 'side','wrist'],
    'policy_class': 'ACT',
    'temporal_agg': False,
    'state_dim': 11,
    'action_dim': 8,
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 2000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}
