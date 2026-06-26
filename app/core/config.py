# app/core/config.py

# Active camera profile switch. Can be set to 'KITTI' or 'ZED_2I'
ACTIVE_CAMERA_PROFILE = 'KITTI'

CAMERA_PROFILES = {
    'KITTI': {
        'fx': 721.5377,  # from typical KITTI calibration (at native_width resolution)
        'fy': 721.5377,
        'cx': 609.5593,
        'cy': 172.854,
        'camera_height_m': 1.65,
        'native_width': 1242,   # calibration reference resolution
        'native_height': 375,
    },
    'ZED_2I': {
        # Using typical ZED 2i HD720 resolution intrinsics
        # These should be replaced with factory calibrated values from your specific ZED 2i
        'fx': 527.3,
        'fy': 527.3,
        'cx': 640.0,
        'cy': 360.0,
        'camera_height_m': 1.65,  # Update based on physical mount height
        'native_width': 1280,
        'native_height': 720,
    }
}

def get_active_intrinsics():
    """Returns the dictionary of camera intrinsics for the active profile (unscaled)."""
    return CAMERA_PROFILES[ACTIVE_CAMERA_PROFILE]


def get_scaled_intrinsics(actual_width: int, actual_height: int = None) -> dict:
    """Return intrinsics scaled to match the actual image resolution.

    The calibration values in CAMERA_PROFILES are defined at a specific
    ``native_width``.  When the image is resized (e.g. 1242 → 640), the
    focal lengths and principal point must be scaled proportionally so that
    the 3-D back-projection math stays correct.

    This is the same scaling that ``bev_projection.intrinsics_from_frame_width``
    applies, ensuring OGM and BEV use identical camera geometry.

    Args:
        actual_width: Pixel width of the image being processed.
        actual_height: Pixel height (optional, derived from aspect ratio).

    Returns:
        Scaled intrinsics dict with keys 'fx', 'fy', 'cx', 'cy',
        'camera_height_m'.
    """
    profile = CAMERA_PROFILES[ACTIVE_CAMERA_PROFILE]
    native_w = profile['native_width']
    scale = actual_width / native_w

    return {
        'fx': profile['fx'] * scale,
        'fy': profile['fy'] * scale,
        'cx': profile['cx'] * scale,
        'cy': profile['cy'] * scale,
        'camera_height_m': profile['camera_height_m'],
    }
