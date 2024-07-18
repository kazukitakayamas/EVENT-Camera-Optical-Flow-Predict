import torch
from torch import nn
from src.models.base import RAFT  # RAFTをimport
from typing import Dict, Any

class EVFlowNet(nn.Module):
    def __init__(self, args: Dict[str, Any]):
        """
        Initializes the EVFlowNet model.

        Args:
            args (Dict[str, Any]): Configuration dictionary for the model.
        """
        super(EVFlowNet, self).__init__()
        self._args = args

        # Create an instance of RAFT
        self.raft = RAFT(args)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, event_volume: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            event_volume (torch.Tensor): Input tensor containing the event data.

        Returns:
            torch.Tensor: Predicted optical flow.
        """
        # Optical flow estimation using RAFT
        flow_predictions = self.raft(event_volume)
        return flow_predictions[-1]  # Return the final flow map
