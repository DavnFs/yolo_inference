import torch
import sys

# Mock Ultralytics custom components that the checkpoint might need
class MockModule(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

# Inject into sys.modules so torch.load finds them
sys.modules['ultralytics.nn.modules'] = type('Mock', (), {'conv': type('Mock', (), {'ASPP': MockModule})})
import ultralytics.nn.modules.conv
ultralytics.nn.modules.conv.ASPP = MockModule

class DualHead(MockModule): pass
class CBAM(MockModule): pass
class SimAM(MockModule): pass

import builtins
builtins.ASPP = MockModule
builtins.CBAM = DualHead
builtins.SimAM = SimAM
builtins.DualHead = DualHead

try:
    ckpt = torch.load('convert/best.pt', map_location='cpu', weights_only=False)
    print("KEYS:", ckpt.keys())
    if 'model' in ckpt:
        print("NAMES:", getattr(ckpt['model'], 'names', 'NO NAMES ATTR'))
except Exception as e:
    print("FAILED:", e)
