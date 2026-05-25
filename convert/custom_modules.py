import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("../app/models/yolov11_SimAM_ASPP.onnx", providers=["CPUExecutionProvider"])
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
out = session.run(None, {session.get_inputs()[0].name: dummy})
print(out[0].shape)  # Should print (1, 19, 8400)