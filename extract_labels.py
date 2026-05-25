import sys
import torch
import warnings
warnings.filterwarnings("ignore")

class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
class _Unpickler(torch.serialization._get_default_unpickler()):
    def find_class(self, module, name):
        if name in ('ASPP', 'CBAM', 'SimAM', 'DualHead'):
            return DummyModule
        try:
            return super().find_class(module, name)
        except Exception:
            return DummyModule

def custom_load(f):
    with open(f, 'rb') as file:
        return torch.serialization._legacy_load(file, map_location='cpu', pickle_module=sys.modules[__name__])

sys.modules[__name__].Unpickler = _Unpickler

try:
    ckpt = torch.load('convert/best.pt', map_location='cpu', weights_only=False, pickle_module=sys.modules[__name__])
    print(ckpt['model'].names)
except Exception as e:
    import builtins
    builtins.ASPP = DummyModule
    builtins.CBAM = DummyModule
    try:
        ckpt = torch.load('convert/best.pt', map_location='cpu', weights_only=False)
        print(ckpt['model'].names)
    except Exception as e:
        print("Failure", e)
