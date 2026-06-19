import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class SimAM(nn.Module):
    def __init__(self, c1, c2=None, e_lambda=1e-4):
        super().__init__()
    def forward(self, x): return x

class ASPP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x): return x
    
class FPN(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x): return x

class PANet(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x): return x

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
modules.SimAM = SimAM
modules.ASPP = ASPP
tasks.SimAM = SimAM
tasks.ASPP = ASPP
modules.FPN = FPN
modules.PANet = PANet
tasks.FPN = FPN
tasks.PANet = PANet

# If there's CBAM or DualHead, just mock it
class CBAM(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x): return x
modules.CBAM = CBAM
tasks.CBAM = CBAM

from ultralytics import YOLO

model = YOLO("convert/best.pt")
print("EXTRACTED_NAMES:", model.names)
