import os 
from utils import merge_eval_statistics

if __name__=="__main__":
    MODEL_NAME='DDIM'
    ENV_NAME='can'
    path_1='visualize/0_Models/DiffusionModel/can/25-04-16-20-33-20/eval_statistics.npz'
    #'visualize/0_Models/DiffusionModel/walker2d-medium-v2/25-04-16-00-31-10-DDPM/eval_statistics.npz'
    path_2='visualize/0_Models/DiffusionModel/can/25-04-17-11-13-29/eval_statistics.npz'
    #'visualize/0_Models/DiffusionModel/walker2d-medium-v2/25-04-16-13-45-24-DDPM2/eval_statistics.npz'
    out_dir=f'visualize/{MODEL_NAME}/{ENV_NAME}/25-04-17-21-12-00-DDIM-can/'
    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir, 'eval_statistics.npz')
    merged_stats=merge_eval_statistics(
        path_1,
        path_2,
        out_path
    )
    print(f"merged_stats={merged_stats}")