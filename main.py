from google.colab import drive
drive.mount('/content/drive')

#ここのインストールは二回実行する必要がある
!pip install hydra-core
!pip install omegaconf
!pip install hdf5plugin
!pip install pytorch-optimizer



import os, sys
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np

# sys.path.appendを最初に配置
sys.path.append('/content/drive/MyDrive/2024 DL Basic/自習用/最終課題/event-camera/dl_lecture_competition_pub')

from src.models.evflownet_1 import EVFlowNet
from src.datasets_3 import DatasetProvider
from enum import Enum, auto
from src.datasets_3 import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR  # CosineAnnealingLRを使用
from pytorch_optimizer import Lookahead  # pytorch-optimizerからLookaheadをインポート
from src.utils import RepresentationType, VoxelGrid, flow_16bit_to_float
import pytorch_optimizer as optim
import matplotlib.pyplot as plt



class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    np.save(f"{file_name}.npy", flow.cpu().numpy())

def plot_optical_flow(original_image, predicted_flow, index):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot the original event image
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Event Image')
    axs[0].axis('off')
    
    # Plot the predicted optical flow
    if isinstance(predicted_flow, np.ndarray):
        predicted_flow = torch.tensor(predicted_flow)
    flow_magnitude = torch.norm(predicted_flow, p=2, dim=0).cpu().numpy()
    axs[1].imshow(flow_magnitude, cmap='inferno')
    axs[1].set_title('Predicted Optical Flow')
    axs[1].axis('off')
    
    plt.show()

@hydra.main(version_base=None, config_path="/content/drive/MyDrive/2024 DL Basic/自習用/最終課題/event-camera/dl_lecture_competition_pub/configs/", config_name="base.yaml")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader setup
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=args.data_loader.common.num_voxel_bins
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set, batch_size=args.data_loader.train.batch_size, shuffle=args.data_loader.train.shuffle, collate_fn=collate_fn, drop_last=False)
    test_data = DataLoader(test_set, batch_size=args.data_loader.test.batch_size, shuffle=args.data_loader.test.shuffle, collate_fn=collate_fn, drop_last=False)

    # Model setup
    model = EVFlowNet(args.train).to(device)

    # 事前学習モデルのロード
    if args.train.pretrained_model_path is not None:
        model.load_state_dict(torch.load(args.train.pretrained_model_path, map_location=device))
        print(f"Pretrained model loaded from {args.train.pretrained_model_path}")

    # Optimizer setup
    base_optimizer = optim.RAdam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)   # Lookaheadの適用

    # Learning rate scheduler setup
    scheduler = CosineAnnealingLR(optimizer, T_max=args.train.epochs)

    # Adding a layer to adjust the input channels if needed
    input_adjust_layer = nn.Conv2d(in_channels=15, out_channels=4, kernel_size=1).to(device)

    # Training loop
    model.train()
    for epoch in range(args.train.epochs):
        total_loss = 0
        print(f"on epoch: {epoch + 1}")
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            ground_truth_flow = batch["flow_gt"].to(device)

            # Adjust input channels only if needed
            if event_image.shape[1] == 15:
                event_image = input_adjust_layer(event_image)

            flow = model(event_image)
            loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Step the scheduler with the epoch's loss
        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}')

    # Create the directory if it doesn't exist
    model_save_dir = "/content/drive/MyDrive/2024 DL Basic/自習用/最終課題/event-camera/学習モデル/sub"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(model_save_dir, f"model_{current_time}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Predicting
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)

    with torch.no_grad():
        print("start test")
        for i, batch in enumerate(tqdm(test_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)

            # Adjust input channels only if needed
            if event_image.shape[1] == 15:
                event_image = input_adjust_layer(event_image)

            batch_flow = model(event_image)
            flow = torch.cat((flow, batch_flow), dim=0)
            
            # プロットのためのコードを追加
            if i < 5:  # 最初の5バッチのみプロット
                original_image = event_image[0].cpu().numpy().transpose(1, 2, 0)  # バッチの最初の画像
                predicted_flow = batch_flow[0].cpu().numpy().transpose(1, 2, 0)  # バッチの最初の推定フロー
                plot_optical_flow(original_image, predicted_flow, i)
                
        print("test done")

    # Save submission
    file_name = "/content/drive/MyDrive/2024 DL Basic/自習用/最終課題/event-camera/submit_fol/sub/submission_00"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0]]
    main()
