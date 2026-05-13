import torch
import os

def check_checkpoint(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print(f"Checkpoint: {path}")
        print(f"  Round: {checkpoint.get('round')}")
        print(f"  Task ID: {checkpoint.get('task_id')}")
        print(f"  Classes Learned: {checkpoint.get('classes_learned')}")
        if 'model_state_dict' in checkpoint:
            fc_weight = checkpoint['model_state_dict'].get('fc.weight')
            if fc_weight is not None:
                print(f"  FC Weight Shape: {fc_weight.shape}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

check_checkpoint(r"c:\Users\LENOVO\Desktop\glfc\sim-glfc-gpu\ketquatrain180round\checkpoint_latest.pt")
check_checkpoint(r"c:\Users\LENOVO\Desktop\glfc\sim-glfc-gpu\ketquatrain180round\checkpoint_latest_1-79.pt")
