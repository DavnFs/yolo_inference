# app/core/config.py

ACTIVE_CAMERA_PROFILE = "KITTI"  # 'KITTI' or 'ZED_2I'

CAMERA_PROFILES = {
    "KITTI": {
        "fx": 721.5377,
        "fy": 721.5377,
        "cx": 609.5593,
        "cy": 172.854,
        "camera_height_m": 1.65,
        "native_width": 1242,
        "native_height": 375,
    },
    "ZED_2I": {
        # Placeholder values — replace with factory-calibrated intrinsics
        # for your specific unit before trusting distance outputs.
        "fx": 527.3,
        "fy": 527.3,
        "cx": 640.0,
        "cy": 360.0,
        "camera_height_m": 1.65,
        "native_width": 1280,
        "native_height": 720,
    },
}


def get_active_intrinsics() -> dict:
    """Unscaled intrinsics for the active profile."""
    if ACTIVE_CAMERA_PROFILE not in CAMERA_PROFILES:
        raise KeyError(
            f"Unknown camera profile '{ACTIVE_CAMERA_PROFILE}'. "
            f"Available: {list(CAMERA_PROFILES.keys())}"
        )
    return CAMERA_PROFILES[ACTIVE_CAMERA_PROFILE]


def get_scaled_intrinsics(actual_width: int, actual_height: int = None) -> dict:
    """Scale intrinsics to match the actual frame resolution.

    fx/cx scale with width, fy/cy scale with height. If actual_height
    isn't provided, we fall back to isotropic scaling (scale_y = scale_x),
    which is only correct if the frame's aspect ratio matches the
    calibration reference. Pass actual_height whenever you have it —
    every current caller that skips it is a latent bug, not a valid
    use case.
    """
    profile = get_active_intrinsics()
    native_w = profile["native_width"]
    native_h = profile["native_height"]

    scale_x = actual_width / native_w
    if actual_height is not None:
        scale_y = actual_height / native_h
    else:
        scale_y = scale_x

    return {
        "fx": profile["fx"] * scale_x,
        "fy": profile["fy"] * scale_y,
        "cx": profile["cx"] * scale_x,
        "cy": profile["cy"] * scale_y,
        "camera_height_m": profile["camera_height_m"],
    }
