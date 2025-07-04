# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# MIT License

# Copyright (c) 2024 Intelligent Robot Motion Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ReinFlow Authors adapted this file from DPPO's official repository on 05/28/2025. 
# Changes include: download D4RL official datasets for the Gym tasks, incorporate Humanoid-v2 expert data and 
# pre-trained flow matching policies for all Gym, Kitchen, and Robomimic tasks. 

def get_dataset_download_url(cfg):
    """
    Download processed train.npz and normalization.npz to the dataset paths specified in cfg.
    
    """
    env = cfg.env
    use_d4rl_dataset=cfg.get('use_d4rl_dataset', False)
    # Gym
    if env == "hopper-medium-v2":
        if use_d4rl_dataset:
            return "https://drive.google.com/drive/folders/13QiGNv3-RE9DdmAZ7HvKKb0KRrmGUoyg?usp=drive_link"
        else:
            return "https://drive.google.com/drive/u/1/folders/18Ti-92XVq3sE24K096WAxjC_SCCngeHd"
    elif env == "walker2d-medium-v2":
        if use_d4rl_dataset:
            return "https://drive.google.com/drive/folders/13QiGNv3-RE9DdmAZ7HvKKb0KRrmGUoyg?usp=drive_link"
        else:
            return "https://drive.google.com/drive/u/1/folders/1BJu8NklriunDHsDrLT6fEpcro3_2IPFf"
    elif env == "ant-medium-expert-v2":
        if use_d4rl_dataset:
            return "https://drive.google.com/drive/folders/1dZHv_DxEN3Yukfw5B8LtRQK7nnbmwzKE?usp=drive_link"
        else:
            return  
    elif env == "Humanoid-medium-v3":
        return "https://drive.google.com/drive/folders/1J6nDPwiNRoecn1M8aEagyTKOBhaVJ0Mo?usp=drive_link"  # This is new to DPPO. We collect the data from our own SAC expert agent trained via https://github.com/cubrink/mujoco-2.1-rl-project#
    elif env == "kitchen-complete-v0":
        return "https://drive.google.com/drive/u/1/folders/18aqg7KIv-YNXohTsRR7Zmg-RyDtdhkLc"
    elif env == "kitchen-partial-v0":
        return "https://drive.google.com/drive/u/1/folders/1zLOx1q4FbJK1ZWLui_vhM2x1fMEkBC2D"
    elif env == "kitchen-mixed-v0":
        return "https://drive.google.com/drive/u/1/folders/1HRMM16UC10A00oBqjYOL1E8hS5icwtvo"
    # D3IL
    elif env == "avoid" and cfg.mode == "d56_r12":  # M1
        return "https://drive.google.com/drive/u/1/folders/1ZAPvLQwv2y4Q98UDVKXFT4fvGF5yhD_o"
    elif env == "avoid" and cfg.mode == "d57_r12":  # M2
        return "https://drive.google.com/drive/u/1/folders/1wyJi1Zbnd6JNy4WGszHBH40A0bbl-vkd"
    elif env == "avoid" and cfg.mode == "d58_r12":  # M3
        return "https://drive.google.com/drive/u/1/folders/1mNXCIPnCO_FDBlEj95InA9eWJM2XcEEj"
    # Robomimic-PH
    elif (env == "can" and "ph" in cfg.train_dataset_path and "img" not in cfg.train_dataset_path):
        return "https://drive.google.com/drive/folders/1rpVsdpqWPygL89E-t4SLQmZgwQ3mpNnY?usp=drive_link"
    elif (env == "square" and "ph" in cfg.train_dataset_path and "img" not in cfg.train_dataset_path):
        return "https://drive.google.com/drive/folders/1wqqjT9JZ9LX11l2Sz_vGxfcT3BfcNrGk?usp=drive_link"
    # Robomimic-MH
    elif env == "lift" and "img" not in cfg.train_dataset_path:  # state
        return "https://drive.google.com/drive/u/1/folders/1lbXgMKBTAiFdJqPZqWXpwjEyrVW16MBu"
    elif env == "lift" and "img" in cfg.train_dataset_path:  # img
        return "https://drive.google.com/drive/u/1/folders/1H-UncdzHx6wd5NWVzrQyftfGls7KGz1O"
    elif env == "can" and "img" not in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1J1qSvsDEf40jnMZY9W0r6ww--E3MdmK3"
    elif env == "can" and "img" in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1VGp_5xXXb1-GJutdSc6AZSzXNk-6_vRz"
    elif env == "square" and "img" not in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1mVVNOJ6wt2EXoapF7PKkcqxbsB9gvK-B"
    elif env == "square" and "img" in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1-aGqVeKLIzJCEst8p0ZTjfjkrXFfJLxa"
    elif env == "transport" and "img" not in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1EVHmFx-YdX4MEE1EjwduVayvaH9vvqvK"
    elif env == "transport" and "img" in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1cOkAZQmmETYEPFrnnX0EuD6mv0kUfMO2"
    # Furniture-Bench
    elif env == "one_leg_low_dim":
        return "https://drive.google.com/drive/u/1/folders/1v4LG2D1fS8id5hqE7Jjt7MYNFUEBNyh4"
    elif env == "one_leg_med_dim":
        return "https://drive.google.com/drive/u/1/folders/1ohDuMSgCqGN1CSh1cI_8A0ia3DVHzj3w"
    elif env == "lamp_low_dim":
        return "https://drive.google.com/drive/u/1/folders/14MqDUmuNmTFcBtKw5gcvx7nuir0zmF7V"
    elif env == "lamp_med_dim":
        return "https://drive.google.com/drive/u/1/folders/1bhOoN0xet4ga0rOIvcRHUFoaCjPRNRLf"
    elif env == "round_table_low_dim":
        return "https://drive.google.com/drive/u/1/folders/15oF3qiqGzlT_98FDoTIVtGmLtHd5SIbi"
    elif env == "round_table_med_dim":
        return "https://drive.google.com/drive/u/1/folders/1U27xjdRrKlLC8E33o7jMFZ1HF5P_Soik"
    # unknown
    else:
        raise ValueError(f"Unknown environment {env}")


def get_normalization_download_url(cfg):
    env = cfg.env_name
    use_d4rl_dataset=cfg.get('use_d4rl_dataset', False)
    # Gym
    if env == "hopper-medium-v2":
        if use_d4rl_dataset:
            return "https://drive.google.com/file/d/11q0pwvxdRW7LW0deT1A9GoysZw2ayHtJ/view?usp=drive_link"
        else:
            return "https://drive.google.com/file/d/1HHZ2X6r5io6hjG-MHVFFoPMJV2fJ3lis/view?usp=drive_link"
    elif env == "walker2d-medium-v2":
        if use_d4rl_dataset:
            return "https://drive.google.com/file/d/1tXFFJgAk2u80T-y7k1mTrUr4pYVA1ab7/view?usp=drive_link"
        else:
            return "https://drive.google.com/file/d/1NSX7t3DFKaBj5HNpv91Oo5h6oXTk0zoo/view?usp=drive_link"
    elif env == "ant-medium-expert-v2":
        if use_d4rl_dataset:
            return "https://drive.google.com/file/d/1tgHdWq97SwKxUGh57YbgrH36nx_uKg8T/view?usp=drive_link"
        else:
            return  
    elif env == "Humanoid-medium-v3":
        return "https://drive.google.com/file/d/1i4rlSk5GhO2TBK3n7KYblcu_I1MAg71v/view?usp=drive_link"  # This is new to DPPO. We collect the data from our own SAC expert agent trained via https://github.com/cubrink/mujoco-2.1-rl-project#
    
    elif env == "kitchen-complete-v0":
        return "https://drive.google.com/file/d/1tBATWLoP1E5s08vr5fiUZBzn8EEsjEZh/view?usp=drive_link"
    elif env == "kitchen-partial-v0":
        return "https://drive.google.com/file/d/1Ptt0cwQwmb5_HGNM-zggRaDKfkqqNO5e/view?usp=drive_link"
    elif env == "kitchen-mixed-v0":
        return "https://drive.google.com/file/d/11gj846QTYFPeV14nhcL5Z9OA5RHIGVt1/view?usp=drive_link"
    elif env == "Humanoid-v3":
        if use_d4rl_dataset:
            return "https://drive.google.com/file/d/1i4rlSk5GhO2TBK3n7KYblcu_I1MAg71v/view?usp=sharing"
    # D3IL
    elif env == "avoiding-m5" and cfg.mode == "d56_r12":  # M1
        return "https://drive.google.com/file/d/1PubKaPabbiSdWYpGmouDhYfXp4QwNHFG/view?usp=drive_link"
    elif env == "avoiding-m5" and cfg.mode == "d57_r12":  # M2
        return "https://drive.google.com/file/d/1Hoohw8buhsLzXoqivMA6IzKS5Izlj07_/view?usp=drive_link"
    elif env == "avoiding-m5" and cfg.mode == "d58_r12":  # M3
        return "https://drive.google.com/file/d/1qt7apV52C9Tflsc-A55J6uDMHzaFa1wN/view?usp=drive_link"
    # Robomimic-PH
    elif (
        env == "can"
        and "ph" in cfg.normalization_path
        and "img" not in cfg.normalization_path
    ):
        return "https://drive.google.com/file/d/1y04FAEXgK6UlZuDiQzTumS9lz-Ufn47B/view?usp=drive_link"
    elif (
        env == "square"
        and "ph" in cfg.normalization_path
        and "img" not in cfg.normalization_path
    ):
        return "https://drive.google.com/file/d/1_75UM0frCZVtcROgfWsdJ0FstToZd1b5/view?usp=drive_link"
    # Robomimic-MH
    elif env == "lift" and "img" not in cfg.normalization_path:  # state
        return "https://drive.google.com/file/d/1d3WjwRds-7I5bBFpZuY27OT9ycb8r_QM/view?usp=drive_link"
    elif env == "lift" and "img" in cfg.normalization_path:  # img
        return "https://drive.google.com/file/d/15GnKDIK8VasvUHahcvEeK_uEs1J9i0ja/view?usp=drive_link"
    elif env == "can" and "img" not in cfg.normalization_path:
        return "https://drive.google.com/file/d/14FxHk9zQ-5ulAO26a6xrdvc-gRkL36FR/view?usp=drive_link"
    elif env == "can" and "img" in cfg.normalization_path:
        return "https://drive.google.com/file/d/1APAB6W10ECaNVVL72F0C2wC6oJhhbjWX/view?usp=drive_link"
    elif env == "square" and "img" not in cfg.normalization_path:
        return "https://drive.google.com/file/d/1FFMqWVv0145OJjbA_iglkWywmdK22Za-/view?usp=drive_link"
    elif env == "square" and "img" in cfg.normalization_path:
        return "https://drive.google.com/file/d/1jq5atfHdu-ZMQ8YaFjwcctbJESBYDgP8/view?usp=drive_link"
    elif env == "transport" and "img" not in cfg.normalization_path:
        return "https://drive.google.com/file/d/1EmC80gIgLoqQ8kRPH5r0mDVqX6NqHO3p/view?usp=drive_link"
    elif env == "transport" and "img" in cfg.normalization_path:
        return "https://drive.google.com/file/d/1LBgvIacNzbXCZXWKYanddqiotevG3hmA/view?usp=drive_link"
    # Furniture-Bench
    elif env == "one_leg_low_dim":
        return "https://drive.google.com/file/d/1fbYxau8Z0tifeuu_06UKdRRz8zozxMUE/view?usp=drive_link"
    elif env == "one_leg_med_dim":
        return "https://drive.google.com/file/d/1bT-MIG99-uwSXL3VwCz6kK9Kg9t6a7BZ/view?usp=drive_link"
    elif env == "lamp_low_dim":
        return "https://drive.google.com/file/d/1KRyCcCT63-cikz6Q_McXM0iVYx9M2E_o/view?usp=drive_link"
    elif env == "lamp_med_dim":
        return "https://drive.google.com/file/d/1WBHKbA6BtSfF3qfRIkyuWga5T9X3fEvg/view?usp=drive_link"
    elif env == "round_table_low_dim":
        return "https://drive.google.com/file/d/1CwfNLI5KXEkl_jVgtAY7-xfq7jED-xc0/view?usp=drive_link"
    elif env == "round_table_med_dim":
        return "https://drive.google.com/file/d/1vNxzok6f6HROGxQR2eoThsN62hVjpayn/view?usp=drive_link"
    # unknown
    else:
        raise ValueError(f"Unknown environment {env}")


def get_checkpoint_download_url(cfg):
    path = cfg.base_policy_path
    use_d4rl_dataset=cfg.get('use_d4rl_dataset', False)
    ######################################
    ####             Gym
    ######################################
    # Hopper (DPPO data)
    if (
        "hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-10-05/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1uV3beg2YuBRh11t7jnRsyd_M9bsFGOkA/view?usp=drive_link"
    elif(
        "pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_42/checkpoint/state_1500.pt"
        in path  # 1-ReFlow
    ):
        return "https://drive.google.com/file/d/1ZehJ_3Qk-DWyZiA5_crJt2b4Im6379_Q/view?usp=drive_link"
    elif(
        "pretrain/hopper-medium-v2_pre_shortcut_mlp_ta4_td20/2025-04-25_08-57-19_42/state_40.pt"
        in path  # Shortcut
    ):
        return "https://drive.google.com/file/d/146oIFc3Md4zqxekAAwUQFuu8mfaBI7w2/view?usp=drive_link"
    # Hopper (D4RL data)
    elif(
        "pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-10-05_D4RL/checkpoint/state_120.pt"
        in path  # DDPM
    ):
        return "https://drive.google.com/file/d/1V9_exmvAyq4jH8nKZJxho5fE7WVAARMm/view?usp=drive_link"
    elif(
        "pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt"
        in path  # 1-ReFlow
    ):
        return "https://drive.google.com/file/d/1AIqtm1ii5euFMlZltl0V5b02zwpP1W8Z/view?usp=drive_link"
    elif(
        "pretrain/hopper-medium-v2_pre_shortcut_mlp_ta4_td20/2025-04-25_08-57-19_D4RL_42/state_40.pt"
        in path  # Shortcut
    ):
        return "https://drive.google.com/file/d/1No8BXrwHILQly0728df6hld0tlf1TEjS/view?usp=drive_link"
    # Walker2d (DPPO data)
    elif (
        "walker2d-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-06-12/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1StopetttozWba4l9u0VT_JW1rYe6kJx2/view?usp=drive_link"
    elif (
        "pretrain/walker2d-v2/ReFlow/2025-02-06_01-39-14_42/checkpoint/state_1500.pt"
        in path # 1-ReFlow
    ):
        return "https://drive.google.com/file/d/1xm4TMel5N_kkABi3VwDJQIqFBSMmbzTj/view?usp=drive_link"
    elif (
        "pretrain/walker2d-v2/ShortCut/2025-04-25_12-55-43_42/checkpoint/state_40.pt"
        in path # Shortcut
    ):
        return "https://drive.google.com/file/d/1YYw8kZjj9lcOtPGH9jDP1RPpjDaDAUKV/view?usp=drive_link"
    # Walker2d (D4RL data)
    elif (
        "pretrain/walker2d-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-06_D4RL/checkpoint/state_60.pt"
        in path # DDPM
    ):
        return "https://drive.google.com/file/d/1DMXfqq3a92QxPQKbtrJKoBgDKRzUhumj/view?usp=drive_link"
    elif (
        "pretrain/walker2d-v2/ReFlow/2025-02-06_01-39-14_D4RL/checkpoint/state_80.pt"
        in path # 1-ReFlow
    ):
        return "https://drive.google.com/file/d/1FKxAtu0tYzplg4Ky-0_iLWinhJ2l0HiW/view?usp=drive_link"
    elif (
        "pretrain/walker2d-v2/ShortCut/2025-04-25_12-55-43_D4RL_42/checkpoint/state_40.pt"
        in path # Shortcut
    ):
        return "https://drive.google.com/file/d/1GwUdWcp8sBy1L8M0n3wW9ngwsYlabFZI/view?usp=drive_link"
    # Halfcheetah
    elif (
        "halfcheetah-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-04-42/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1o9ryyeZQAsaB4ffUTCJkIaGCi0frL3G4/view?usp=drive_link"
    # Ant-v0 (Sensitivity Analysis)
    elif (
        "pretrain/ant-medium-expert-v0_pre_diffusion_mlp_ta4_td20/2025-03-29_20-19-29_42/checkpoint/state_6_700.pt"
        in path # DDPM
    ):
        return "https://drive.google.com/file/d/1rW0ylPMBAlG5U9XTHkOnNRZmnUZ71fyj/view?usp=drive_link"
    elif (
        "pretrain/ant-v0/ReFlow/2025-03-29_19-23-57_42/checkpoint/best_769.pt"
        in path # 1-ReFlow
    ):
        return "https://drive.google.com/file/d/17xvfI9WValAAMIUG-vIi2rsKsCoreFCR/view?usp=drive_link"
    elif (
        "pretrain/ant-v0/ShortCut/2025-04-25_08-57-42_42/checkpoint/state_27.pt"
        in path # Shortcut
    ):
        return "https://drive.google.com/file/d/1brdcFfRxYBp2HA_yH9a0bN69E7dkR8hx/view?usp=drive_link"
    # Ant-v2 (D4RL data)
    elif (
        "pretrain/ant-medium-expert-v0_pre_diffusion_mlp_ta4_td20/2025-03-29_20-19-29_42_D4RL/checkpoint/state_20.pt"
        in path   # DDPM
    ):
        return "https://drive.google.com/file/d/10C5uLwkibOMiVLL7sCzw-y-KDgYFInRK/view?usp=drive_link"
    elif (
        "pretrain/ant-v2/ReFlow/2025-03-29_19-23-57_D4RL_42/checkpoint/state_50.pt" # 1-ReFlow
        in path
    ):
        return "https://drive.google.com/file/d/1xbdcgA6qmQhZR9_O0uf3vhwq0oaXBEgx/view?usp=drive_link"
    elif (
        "pretrain/ant-v2/ShortCut/2025-04-25_08-57-42_D4RL_42/checkpoint/state_50.pt" # Shortcut
        in path
    ):
        return "https://drive.google.com/file/d/173OpIy19xHY0DBWTkUr0RD7KJJrRP5ra/view?usp=drive_link"
    # Humanoid-v2 (Sensitivity Analysis)
    elif (
        "pretrain/Humanoid-medium-v3_pre_diffusion_mlp_ta4_td20/2025-05-01_19-04-33_42/checkpoint/state_40.pt"
        in path   # DDPM
    ):
        return "https://drive.google.com/file/d/1UAnQxw1Xq-gggfF03tju7EwxTHfni1fy/view?usp=drive_link"
    elif (
        "pretrain/Humanoid-v3_pre_reflow_mlp_ta4_td20/2025-05-01_18-18-08_42/checkpoint/state_50.pt" # 1-ReFlow
        in path
    ):
        return "https://drive.google.com/file/d/1IWNClGO-619zCJA5WFJX6PbDmQ0w1zfl/view?usp=drive_link"
    elif (
        "pretrain/Humanoid-medium-v3_pre_shortcut_mlp_ta4_td20/2025-05-01_18-18-08_42/checkpoint/state_30.pt" # Shortcut
        in path
    ):
        return "https://drive.google.com/file/d/1pboukTyc3Jk_xqhzld0rZF4ZJPZ3ho_o/view?usp=drive_link"
    # Humanoid-v2 (D4RL data)
    elif (
        "pretrain/Humanoid-medium-v3_pre_diffusion_mlp_ta4_td20/2025-05-01_19-04-33_D4RL_42/checkpoint/state_120.pt"
        in path   # DDPM
    ):
        return "https://drive.google.com/file/d/1jPXmZ-jaxKXSjUE3WjofU6nDncpH7SEz/view?usp=drive_link"
    elif (
        "pretrain/Humanoid-v3_pre_reflow_mlp_ta4_td20/2025-05-01_18-18-08_42_D4RL/checkpoint/state_100.pt" # 1-ReFlow
        in path
    ):
        return "https://drive.google.com/file/d/1StJeZrgjsY0mdEDR5JsXXj9aEAma5Jyt/view?usp=drive_link"
    elif (
        "pretrain/Humanoid-medium-v3_pre_shortcut_mlp_ta4_td20/2025-05-01_18-18-08_42_D4RL/checkpoint/state_60.pt" # Shortcut
        in path
    ):
        return "https://drive.google.com/file/d/1-VSl094U6zZ3hcYqR3-YugDBiyiEAdkt/view?usp=drive_link"
    # Demo-RL
    elif (
        "halfcheetah-medium-v2_pre_diffusion_mlp_ta1_td20/2024-09-29_02-13-10_42/checkpoint/state_1000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Oi5JhsU45ScHdlrtn5AX8Ji7InLBVj4D/view?usp=drive_link"
    elif (
        "halfcheetah-medium-v2_pre_gaussian_mlp_ta1/2024-09-28_18-48-54_42/checkpoint/state_500.pt"
        in path
    ):
        return "https://drive.google.com/file/d/14rbYGaCxvj1PtELKVfdXNHJ1Od2G6FLw/view?usp=drive_link"
    elif (
        "halfcheetah-medium-v2_calql_mlp_ta1/2024-09-29_22-59-08_42/checkpoint/state_49.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Xf758xzsAqpFwV955OVUNL6Za90XPo1K/view?usp=drive_link"
    elif (
        "kitchen-complete-v0_pre_diffusion_mlp_ta4_td20/2024-10-20_16-47-42_42/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1YBwyNd30a4_inu2sZzNSNLJQsj8fN3ZX/view?usp=drive_link"
    elif (
        "kitchen-complete-v0_calql_mlp_ta1/2024-10-26_01-01-33_42/checkpoint/state_999.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1K4V59iXNQbpOvu3u5y6C9R5piMU9idYm/view?usp=drive_link"
    elif (
        "kitchen-complete-v0_pre_gaussian_mlp_ta1/2024-10-25_14-48-43_42/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1tQYgnkdhR5wnuXC4Ha_mKHuIdg6J627s/view?usp=drive_link"
    elif (
        "kitchen-complete-v0_pre_reflow_mish_mlp_ta4_td4/2025-05-05_20-13-17_42/checkpoint/state_500.pt"  # 1-ReFlow Policy for kitchen-complete-v0
        in path
    ):
        return "https://drive.google.com/file/d/1tNH1z9eb0urIDwOzx2EBFiWIzovRHqZ7/view?usp=drive_link"
    elif (
        "kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt"  # Shortcut Policy for kitchen-complete-v0
        in path
    ):
        return "https://drive.google.com/file/d/1zyJMwVKPbzX8AJheH2W0qZ5VLjeHvabi/view?usp=drive_link"
    elif (
        "kitchen-partial-v0_pre_diffusion_mlp_ta4_td20/2024-10-20_16-48-29_42/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1oSupKkUjCFQVWBIJV5Seh-CclWhgpopS/view?usp=drive_link"
    elif (
        "kitchen-partial-v0_calql_mlp_ta1/2024-10-25_21-26-51_42/checkpoint/state_980.pt"
        in path
    ):
        return "https://drive.google.com/file/d/17HUDp3l8mJsMIW-DRraKPhUkH44KGTbA/view?usp=drive_link"
    elif (
        "kitchen-partial-v0_pre_gaussian_mlp_ta1/2024-10-25_01-45-52_42/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1-ZmGRPi4jMS7HfqHPvWrSPxNSoTwih6q/view?usp=drive_link"
    elif (
        "pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt"  # Shortcut Policy for kitchen-partial-v0
        in path
    ):
        return "https://drive.google.com/file/d/1FuU66a4hnwA4JKsOHOQ5aq_hg17zEyA0/view?usp=drive_link"
    elif (
        "kitchen-mixed-v0_pre_diffusion_mlp_ta4_td20/2024-10-20_16-48-28_42/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1X24Hqbn4b4xyLK_1A3D6zhSgsN7frVCG/view?usp=drive_link"
    elif (
        "kitchen-mixed-v0_calql_mlp_ta1/2024-10-25_21-36-13_42/checkpoint/state_999.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1AP7bbzAwwfuSLmV1HkQLfmd76MXQn2Za/view?usp=drive_link"
    elif (
        "kitchen-mixed-v0_pre_gaussian_mlp_ta1/2024-10-25_01-39-44_42/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1LEzGhMOqL3YZFXMGn1mTcOh-tm4Lh1SH/view?usp=drive_link"
    elif (
        "pretrain/kitchen-mixed-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-11-00_42/checkpoint/state_2400.pt"  # Shortcut Policy for kitchen-mixed-v0
        in path
    ):
        return "https://drive.google.com/file/d/1cl60hKo6nWdmNHkGJ1jqYDgOHtt4toZz/view?usp=drive_link"
    ######################################
    ####             D3IL
    ######################################
    elif (
        "avoid_d56_r12_pre_diffusion_mlp_ta4_td20/2024-07-06_22-50-07/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1JdEOG0KsCA9EX9zq09DE0xkTB4xy6DNp/view?usp=drive_link"
    elif (
        "avoid_d56_r12_pre_gaussian_mlp_ta4/2024-07-07_01-35-48/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/138wEi_rVV5HpcwgH6_3BlXQ1dhgZN05L/view?usp=drive_link"
    elif (
        "avoid_d56_r12_pre_gmm_mlp_ta4/2024-07-10_14-30-00/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1krvEINP6UBfnJG9bEge53h3MNJkTQXe_/view?usp=drive_link"
    elif (
        "avoid_d57_r12_pre_diffusion_mlp_ta4_td20/2024-07-07_13-12-09/checkpoint/state_15000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1wAmRHzRZ4O5z_ZWDhVg5JFZbugo4cHKA/view?usp=drive_link"
    elif (
        "avoid_d57_r12_pre_gaussian_mlp_ta4/2024-07-07_02-15-50/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1hf_647bJ0EMRhArsfxStSkhgkGaWGhmD/view?usp=drive_link"
    elif (
        "avoid_d57_r12_pre_gmm_mlp_ta4/2024-07-10_15-44-32/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1CE4AcNJp2UITIHpuLUwxp7cDH8Y8jsYC/view?usp=drive_link"
    elif (
        "avoid_d58_r12_pre_diffusion_mlp_ta4_td20/2024-07-07_13-54-54/checkpoint/state_15000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1w5X1lJZd0wI6E2XRdj839TqJ_9EziGRx/view?usp=drive_link"
    elif (
        "avoid_d58_r12_pre_gaussian_mlp_ta4/2024-07-07_13-11-49/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1YIVEN0Ykica9dj_DxxPrB4QJLgv-Ne0N/view?usp=drive_link"
    elif (
        "avoid_d58_r12_pre_gmm_mlp_ta4/2024-07-10_17-01-50/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/174tadrqjfxJdOgsMjNbg53goZtIkpW3f/view?usp=drive_link"
    ######################################
    ####             Robomimic-Lift
    ######################################
    elif (
        "lift_pre_diffusion_unet_ta4_td20/2024-06-29_02-49-45/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1T-NGgBmT-UmcVWADygXj873IyWLewvsU/view?usp=drive_link"
    elif (
        "lift_pre_diffusion_mlp_ta4_td20/2024-06-28_14-47-58/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Ngr-DNxoB9XNCZ2O-NF5p60NzmYlzmWG/view?usp=drive_link"
    elif (
        "lift_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-24-35/checkpoint/state_2500.pt"
        in path
    ):
        return "https://drive.google.com/file/d/19hqNicwKKKDrlS5UMr-FLRu51EAW8Z51/view?usp=drive_link"
    elif (
        "lift_pre_gaussian_mlp_ta4/2024-06-28_14-48-24/checkpoint/state_5000.pt" in path
    ):
        return "https://drive.google.com/file/d/157x5_XJy3ZyaPz_7Vr_opXAZ_mPvlmhp/view?usp=drive_link"
    elif (
        "lift_pre_gaussian_mlp_img_ta4/2024-07-28_23-00-48/checkpoint/state_500.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Uae7K3Hv9XzaAGljG2fjsxYyOEgL03zv/view?usp=drive_link"
    elif (
        "lift_pre_gaussian_transformer_ta4/2024-06-28_14-49-23/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Z_C8ureDDPXqpDUaRMVKBt3VgfZlIQdE/view?usp=drive_link"
    elif "lift_pre_gmm_mlp_ta4/2024-06-28_15-30-32/checkpoint/state_5000.pt" in path:
        return "https://drive.google.com/file/d/1wFvBoIOaCJVibqEIYSjJxAbxjGkH7RMk/view?usp=drive_link"
    elif (
        "lift_pre_gmm_transformer_ta4/2024-06-28_14-51-23/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1w_3WOXS51debWc1ShgaO2VQx49dj9ky2/view?usp=drive_link"
    ######################################
    ####             Robomimic-Can
    ######################################
    elif (
        "can_pre_diffusion_unet_ta4_td20/2024-06-29_02-49-45/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1s346KCe2aar_tXX7u8rzjRF3kpwVpH5c/view?usp=drive_link"
    elif (
        "can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1L1ZLD1u1Y1YJmRLGzScXbQ02wGS-_cWo/view?usp=drive_link"
    elif (
        "can_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-23-55/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1siIKDGVHld3ZH8vDqgu9iE5H9KoaYq_Z/view?usp=drive_link"
    elif (
        "can/ReFlow/state_2000.pt"   # 1-ReFlow for can-img
        in path
    ):
        return "https://drive.google.com/file/d/1e2OwtcbJe_G2XcXLisEuJ-YFt7kJS-23/view?usp=drive_link"
    elif (
        "can/ShortCut/2025-04-25_15-22-25_42_state_2000.pt"   # Shortcut Policy for can-img
        in path
    ):
        return "https://drive.google.com/file/d/1-sdH5z1iz9mmr7l2xq-zZ1zkKf_nPA0G/view?usp=drive_link"
    elif (
        "can_pre_gaussian_mlp_ta4/2024-06-28_13-31-00/checkpoint/state_5000.pt" in path
    ):
        return "https://drive.google.com/file/d/1bA-A0p0KnHwrVO3MqjuYdV3dZuuZICMy/view?usp=drive_link"
    elif (
        "can_pre_gaussian_mlp_img_ta4/2024-07-28_21-54-40/checkpoint/state_1000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/16vUUyVO9DvnDyPtSGiZx4iQP4JxuSAoS/view?usp=drive_link"
    elif (
        "can_pre_gaussian_transformer_ta4/2024-06-28_13-42-20/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1cGf7s8aS5grZsGRb5PYZiPogcPRr_lrf/view?usp=drive_link"
    elif "can_pre_gmm_mlp_ta4/2024-06-28_13-32-19/checkpoint/state_5000.pt" in path:
        return "https://drive.google.com/file/d/1KVx8-KICiIHstcjsvhPlQc_pZZ6RxXpd/view?usp=drive_link"
    elif (
        "can_pre_gmm_transformer_ta4/2024-06-28_13-43-21/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1xSgwGG40zdoO2DDSM79l0rMHeNmaifnq/view?usp=drive_link"
    # demo-PH
    elif (
        "can_pre_diffusion_mlp_ta1_td20/2024-10-14_10-54-33_0/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Ze86hw2E0jJinn3Vx683JQ10Gq5FIJad/view?usp=drive_link"
    elif (
        "can_pre_gaussian_mlp_ta1/2024-10-08_20-52-04_0/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1jP3mEOhZojWiTXCMZ0ajFRMkAAmonGxV/view?usp=drive_link"
    elif "can_calql_mlp_ta1/2024-10-09_11-05-07_0/checkpoint/state_999.pt" in path:
        return "https://drive.google.com/file/d/1ERaZKTXmL-vdyU8PZ2X9GjFIMVKJjA2N/view?usp=drive_link"
    # demo-MH
    elif (
        "can_pre_diffusion_mlp_ta1_td20/2024-09-29_15-43-07_42/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1pEs1cK1x5obAtJA9pFSN1CWG79gNhH24/view?usp=drive_link"
    elif (
        "can_pre_gaussian_mlp_ta1/2024-09-28_13-43-59_42/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Fa3yflkvYSAy6PKT646U1VAqUJ0YHqsj/view?usp=drive_link"
    elif "can_calql_mlp_ta1/2024-10-25_22-30-16_42/checkpoint/state_999.pt" in path:
        return "https://drive.google.com/file/d/1AA94uEaK_SzG2mTpaKqZIwNMh6omL_g0/view?usp=drive_link"
    ######################################
    ####             Robomimic-Square
    ######################################
    elif (
        "square_pre_diffusion_unet_ta4_td20/2024-06-29_02-48-45/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/11IEgQe0LFI23hn1Cwf6Z_YfJdDilVc0z/view?usp=drive_link"
    elif (
        "square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1lP9mNe2AxMigfOywcaHOOR7FxQ-KR_Ee/view?usp=drive_link"
    elif (
        "square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1miud5SX41xjPoW8yqClRhaFx0Q7EWl3V/view?usp=drive_link"
    elif (
        "square/ReFlow/state_2000.pt"   # 1-ReFlow for square-img (uniform time distribution)
        in path
    ):
        return "https://drive.google.com/file/d/1nC4yc9XjXO1YtZmFoWh1No5zn3yhtVeo/view?usp=drive_link"
    elif (
        "square/ShortCut/square_pre_shortcut_mlp_img_ta4_td20/state_2000.pt"   # Shortcut Policy for square-img trained on 100 episodes.
        in path
    ):
        return "https://drive.google.com/file/d/12fWRVDfo3DTxYBkzIJH9R1ahJbP5rFhx/view?usp=drive_link"
    elif (
        "square_pre_gaussian_mlp_ta4/2024-06-28_15-02-32/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1rETiyXLz7YgYoHKwLEZa7dFabu4gfJR8/view?usp=drive_link"
    elif (
        "square_pre_gaussian_mlp_img_ta4/2024-07-30_18-44-32/checkpoint/state_4000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1myB6FOAmt6c6x3ScGKXRULgx826tZVS4/view?usp=drive_link"
    elif (
        "square_pre_gaussian_transformer_ta4/2024-06-28_15-02-39/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1JJQ7KbRWWBB09PLwNRriAUEAl9vjFeCW/view?usp=drive_link"
    elif "square_pre_gmm_mlp_ta4/2024-06-28_15-03-08/checkpoint/state_5000.pt" in path:
        return "https://drive.google.com/file/d/10ujnnOk2Pn-yjE7iW9-RYbApo_aKiNRq/view?usp=drive_link"
    elif (
        "square_pre_gmm_transformer_ta4/2024-06-28_15-03-15/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1LczXhgeNtQfqySsfGNbbviPrlLwyh-E3/view?usp=drive_link"
    # demo-PH
    elif (
        "square_pre_diffusion_mlp_ta1_td20/2024-10-14_10-54-33_0/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1_Jnz14ySxqbtZa9IIEWkXqy5_-EwJLBw/view?usp=drive_link"
    elif (
        "square_pre_gaussian_mlp_ta1/2024-10-08_20-52-42_0/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1ZPWKUoZ93OqqVX3ephQMkpeBZoYrceM5/view?usp=drive_link"
    elif "square_calql_mlp_ta1/2024-10-09_11-05-07_0/checkpoint/state_999.pt" in path:
        return "https://drive.google.com/file/d/1_7YtUwRd_U5tuOvhHogJDhkEsE-4D24V/view?usp=drive_link"
    # demo-MH
    elif (
        "square_pre_diffusion_mlp_ta1_td20/2024-09-29_02-14-14_42/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1ks1PnUBvFVWPnpGnYL8_eIfLNeGZbv1p/view?usp=drive_link"
    elif (
        "square_pre_gaussian_mlp_ta1/2024-09-28_13-42-43_42/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1uIOn8QUkGRbhZLkQ9ziOkP7yGQnpYdk7/view?usp=drive_link"
    elif "square_calql_mlp_ta1/2024-10-25_22-44-12_42/checkpoint/state_999.pt" in path:
        return "https://drive.google.com/file/d/1zgzG6bx6ugAEaq72z9WpXX6iewClcKTV/view?usp=drive_link"
    ######################################
    ####             Robomimic-Transport
    ######################################
    elif (
        "transport_pre_diffusion_unet_ta16_td20/2024-07-04_02-20-53/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1MNGT8j9x1uudugGUcia-xwP_7f7xVY4K/view?usp=drive_link"
    elif (
        "transport_pre_diffusion_mlp_ta8_td20/2024-07-08_11-18-59/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1240FmcDPg_VyXReEtePjBN4MML-OT21C/view?usp=drive_link"
    elif (
        "transport_pre_diffusion_mlp_img_ta8_td100/2024-07-30_22-30-06/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1_UmjNv27_w49XC_EO1WexgreJmC0SEdF/view?usp=drive_link"
    elif (
        "transport/ReFlow/state_2000.pt"   # 1-ReFlow for transport-img (uniform time distribution)
        in path
    ):
        return "" # comming soon ! or you can also train it yourself. 
    elif (
        "transport/ShortCut/state_750.pt"   # Shortcut Policy for transport-img trained on 100 episodes.
        in path
    ):
        return "https://drive.google.com/file/d/1T378Z1-KfgYPSbQCodd9pkKMLsLkvbcM/view?usp=drive_link"
    elif (
        "transport_pre_gaussian_mlp_ta8/2024-07-10_01-50-52/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1NOHDNHu1sTabxBd4DZrSrTyX9l1I2GuO/view?usp=drive_link"
    elif (
        "transport_pre_gaussian_mlp_img_ta8/2024-07-30_21-39-34/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1XYOIwOEgOVoUdMmxWuBRAcA5BOS777Kc/view?usp=drive_link"
    elif (
        "transport_pre_gaussian_transformer_ta8/2024-06-28_15-18-16/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1hnt9lX5bg82iFFsAk5FY3TktPXsAhUFU/view?usp=drive_link"
    elif (
        "transport_pre_gmm_mlp_ta8/2024-07-10_01-51-21/checkpoint/state_5000.pt" in path
    ):
        return "https://drive.google.com/file/d/1da9yLIu5ahq-ZgIIsG7wqehA5VopFTnt/view?usp=drive_link"
    elif (
        "transport_pre_gmm_transformer_ta8/2024-06-28_15-18-43/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1c0S7WX-U1Kn6n-wWZEQSR1alLPW2eZmi/view?usp=drive_link"
    ######################################
    ####             Furniture-One_leg
    ######################################
    elif (
        "one_leg_low_dim_pre_diffusion_mlp_ta8_td100/2024-07-22_20-01-16/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1tP0i53EMwNyw_bS2lt9WtQDIAoTxn40g/view?usp=drive_link"
    elif (
        "one_leg_low_dim_pre_diffusion_unet_ta16_td100/2024-07-03_22-23-38/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/13nOr7EI79RqdRuoc-je0LoTX5Y-IaO9J/view?usp=drive_link"
    elif (
        "one_leg_low_dim_pre_gaussian_mlp_ta8/2024-06-26_23-43-02/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1DeHj-IEX3a2ZLXFlV0MYe1N_5qupod_w/view?usp=drive_link"
    elif (
        "one_leg_med_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-11/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1aEzObl4EkOBKSs0wI2MGmkslzIjZy2_L/view?usp=drive_link"
    elif (
        "one_leg_med_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-16-16/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1pSwp_IUDSQ15OszChCkrrTQYdvG3pHc1/view?usp=drive_link"
    elif (
        "one_leg_med_dim_pre_gaussian_mlp_ta8/2024-06-28_16-27-02/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1lC4PKRPn4tRWR9VW5fSfCu3MKnRvYqD6/view?usp=drive_link"
    ######################################
    ####             Furniture-Lamp
    ######################################
    elif (
        "lamp_low_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-20/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1md5m4vpe5MmStu-fOk8snF-52BYGZpSc/view?usp=drive_link"
    elif (
        "lamp_low_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-16-48/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/103lUuvyKPvp97hzBYnUDAWgor6LsUf21/view?usp=drive_link"
    elif (
        "lamp_low_dim_pre_gaussian_mlp_ta8/2024-06-28_16-26-51/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1tK6NuZf4_xtTlZksIR9oQxiFEADFRVA9/view?usp=drive_link"
    elif (
        "lamp_med_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-20/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1yVUXYCK_vFhQ7mawxnCGxuHS6RgWWDjj/view?usp=drive_link"
    elif (
        "lamp_med_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-17-21/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1_qd47U50on-T1Pqojlfy7S3Cl6bLm_D1/view?usp=drive_link"
    elif (
        "lamp_med_dim_pre_gaussian_mlp_ta8/2024-06-28_16-26-56/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1d16HmHgidtXForqoN5QJg_e152XCaJM5/view?usp=drive_link"
    ######################################
    ####             Furniture-Round_table
    ######################################
    elif (
        "round_table_low_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-26/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1iJyqJr84AtszGqPeSysN70A-SmP73mKu/view?usp=drive_link"
    elif (
        "round_table_low_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-19-48/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1F3RFgLcFemU-IDUrLCXMzmQXknC2ISgJ/view?usp=drive_link"
    elif (
        "round_table_low_dim_pre_gaussian_mlp_ta8/2024-06-28_16-26-51/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1eX2u0cvu_zveblrg2htFomAxSToJmHyL/view?usp=drive_link"
    elif (
        "round_table_med_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-29/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1DDvHEqb7WqVgT3_W6gYCNDQrvGyreatP/view?usp=drive_link"
    elif (
        "round_table_med_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-20-21/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1hzungyvt3Uc-XbrCzVghzdr59zDf6FlD/view?usp=drive_link"
    elif (
        "round_table_med_dim_pre_gaussian_mlp_ta8/2024-06-28_16-26-38/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1LvELZ7A-whxKk1Oq7S8M9uBiSrWPAGQT/view?usp=drive_link"
    # unknown --- this means the user trained own policy but specifies the wrong path
    else:
        return None
