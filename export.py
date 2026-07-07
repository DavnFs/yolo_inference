import numpy as np

# Load the .npy file
data = np.load(
    r"D:\Davin\Kuliah\offline-monit-app\stereo_dataset\kitti_0013\depth_npy\0000000000.npy"
)

# View the array contents
print(data)

# Optional: View metadata like dimensions and data type
print(data.shape)
print(data.dtype)
