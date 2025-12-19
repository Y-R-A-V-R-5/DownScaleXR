"""
Model definitions for DownScaleXR

Design assumptions:
- Downsampling choice is the primary experimental variable
- Architectures are intentionally small and interpretable
- CPU-first execution
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path

class LeNetVariant(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        
        # 1. Load architecture config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.model_name = self.cfg['model_name']
        input_dims = self.cfg['input_dims']  # e.g., [1, 320, 320]

        # Architectural sanity checks
        assert len(input_dims) == 3, "input_dims must be [C, H, W]"
        assert input_dims[1] <= 320, "Input resolution too large for CPU experiments"
        assert input_dims[1] == input_dims[2], "Non-square inputs not supported"

        # 2. Build layers dynamically
        layers = []
        last_out_features = None

        for layer_def in self.cfg["layers"]:
            layer_type = layer_def["type"]
            params = {k: v for k, v in layer_def.items() if k != "type"}

            if layer_type == "Conv2d":
                layers.append(nn.Conv2d(**params))

            elif layer_type == "BatchNorm2d":
                layers.append(nn.BatchNorm2d(**params))

            elif layer_type == "ReLU":
                layers.append(nn.ReLU(inplace=True))

            elif layer_type == "AvgPool2d":
                layers.append(nn.AvgPool2d(**params))

            elif layer_type == "MaxPool2d":
                layers.append(nn.MaxPool2d(**params))

            elif layer_type == "AdaptiveAvgPool2d":
                layers.append(nn.AdaptiveAvgPool2d(**params))

            elif layer_type == "Flatten":
                layers.append(nn.Flatten())

            elif layer_type == "Linear":
                if "in_features" in params:
                    in_f = params["in_features"]
                else:
                    if last_out_features is None:
                        in_f = self._infer_flatten_size(layers, input_dims)
                    else:
                        in_f = last_out_features

                out_f = params["out_features"]
                layers.append(nn.Linear(in_f, out_f))
                last_out_features = out_f

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        self.net = nn.Sequential(*layers)

    def _infer_flatten_size(self, layers, input_dims):
        temp = nn.Sequential(*layers)
        dummy = torch.zeros(1, *input_dims)
        with torch.no_grad():
            out = temp(dummy)

        assert out.numel() > 0, "Invalid downsampling configuration"
        return out.view(1, -1).size(1)

    def forward(self, x):
        return self.net(x) 