import builtins
import sys
import torch

class Dummy:
    pass

class MockModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def __getattr__(self, k):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

sys.modules['ultralytics'] = Dummy()
sys.modules['ultralytics.nn'] = Dummy()
sys.modules['ultralytics.nn.tasks'] = Dummy()
sys.modules['ultralytics.nn.modules'] = Dummy()
sys.modules['ultralytics.nn.modules.conv'] = Dummy()
sys.modules['ultralytics.nn.modules.block'] = Dummy()
sys.modules['ultralytics.nn.modules.head'] = Dummy()

sys.modules['ultralytics.nn.modules.conv'].ASPP = MockModule
sys.modules['ultralytics.nn.modules.block'].CBAM = MockModule

builtins.ASPP = MockModule
builtins.CBAM = MockModule
builtins.SimAM = MockModule
builtins.DualHead = MockModule

try:
    with torch.serialization.safe_globals([MockModule]):
        ckpt = torch.load('convert/best.pt', map_location='cpu', weights_only=False)
        print("NAMES::", ckpt['model'].names)
except Exception as e:
    print("FAILED:", e)
