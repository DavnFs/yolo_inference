"""
ros2_perception_node.py
ROS 2 subscriber node untuk yolo_inference.
Replace StereoSimLoader dengan synchronized subscriber.

Arsitektur:
- Node berjalan di background thread (daemon=True)
- Frame + depth dikirim ke GUI via thread-safe queue.Queue
- GUI tetap di main thread (Tkinter requirement)

Cara integrasi ke gui_main.py:
    runner = ROSBagRunner()
    runner.start()

    # Di processing loop:
    payload = runner.get_frame(timeout=0.1)
    if payload:
        frame_bgr = payload['frame_bgr']   # (H,W,3) uint8
        depth_m   = payload['depth_m']     # (H,W) float32, meter
        calib     = payload['calib']       # dict: fx, fy, cx, cy, width, height

    runner.stop()

Peringatan:
- JANGAN gunakan rclpy.spin() di main thread jika Tkinter aktif.
- message_filters.ApproximateTimeSynchronizer: slop 0.05s cukup untuk bag offline.
  Jika timestamp identik (kita set sama di konverter), switch ke TimeSynchronizer.
- cv_bridge di Jetson: INSTALL via apt (ros-humble-cv-bridge), BUKAN pip.
"""

import threading
import queue
from typing import Optional

import numpy as np

_ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image, CameraInfo
    import message_filters
    from cv_bridge import CvBridge
    _ROS2_AVAILABLE = True
except ImportError:
    pass  # Graceful degradation if ROS2 not installed


TOPIC_RGB   = '/kitti/camera/left/image_raw'
TOPIC_DEPTH = '/kitti/camera/left/depth'
TOPIC_INFO  = '/kitti/camera/left/camera_info'


class PerceptionSubscriberNode:
    """
    ROS 2 node yang subscribe RGB + Depth secara tersinkronisasi.
    Hanya diinstansiasi jika _ROS2_AVAILABLE = True.
    """

    def __init__(self, frame_queue: queue.Queue):
        if not _ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 Python packages tidak tersedia. "
                               "Jalankan: source /opt/ros/humble/setup.bash")

        # Lazy import di dalam class untuk menghindari NameError
        from rclpy.node import Node
        import message_filters
        from cv_bridge import CvBridge

        class _InnerNode(Node):
            def __init__(inner_self, outer):
                super().__init__('yolo_perception_subscriber')
                inner_self._bridge = CvBridge()
                inner_self._frame_queue = frame_queue
                inner_self._calib_cache = None

                inner_self._info_sub = inner_self.create_subscription(
                    CameraInfo,
                    TOPIC_INFO,
                    lambda msg: outer._camera_info_callback(msg, inner_self),
                    qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
                )

                rgb_sub = message_filters.Subscriber(
                    inner_self, Image, TOPIC_RGB,
                    qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
                )
                depth_sub = message_filters.Subscriber(
                    inner_self, Image, TOPIC_DEPTH,
                    qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
                )

                # Gunakan TimeSynchronizer karena konverter kita set timestamp identik
                inner_self._sync = message_filters.TimeSynchronizer(
                    [rgb_sub, depth_sub], queue_size=10
                )
                inner_self._sync.registerCallback(
                    lambda r, d: outer._synced_callback(r, d, inner_self)
                )
                inner_self.get_logger().info('PerceptionSubscriberNode ready.')

        self._node = _InnerNode(self)
        self._frame_queue = frame_queue

    def _camera_info_callback(self, msg, node):
        if node._calib_cache is not None:
            return
        K = msg.k
        node._calib_cache = {
            'fx': K[0],
            'fy': K[4],
            'cx': K[2],
            'cy': K[5],
            'width': msg.width,
            'height': msg.height,
        }
        node.get_logger().info(
            f'CameraInfo cached: fx={K[0]:.2f}, fy={K[4]:.2f}, '
            f'cx={K[2]:.2f}, cy={K[5]:.2f}'
        )

    def _synced_callback(self, rgb_msg, depth_msg, node):
        """
        Konversi dilakukan di background thread agar GUI tidak terbebani.
        Queue.put_nowait() agar tidak blocking — drop frame jika penuh.
        """
        try:
            frame_bgr = node._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # 32FC1 → float32 numpy, sudah dalam meter dari konverter kita
            depth_m = node._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth_m.dtype != np.float32:
                depth_m = depth_m.astype(np.float32)

            payload = {
                'frame_bgr': frame_bgr,
                'depth_m': depth_m,
                'calib': node._calib_cache,
                'timestamp': rgb_msg.header.stamp,
            }
            try:
                self._frame_queue.put_nowait(payload)
            except queue.Full:
                pass  # GUI lebih lambat dari bag — drop frame, tidak perlu log
        except Exception as e:
            node.get_logger().error(f'Synced callback error: {e}')

    def get_ros_node(self):
        return self._node


class ROSBagRunner:
    """
    Wrapper untuk menjalankan ROS 2 node di background thread.
    Integrasikan ke DashboardGUI sebagai pengganti StereoSimLoader.

    Queue maxsize=2: jika GUI lambat, frame baru menimpa yang lama.
    Lebih baik drop frame daripada accumulate lag.
    """

    def __init__(self):
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._perception_node: Optional[PerceptionSubscriberNode] = None
        self._executor = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def is_available(self) -> bool:
        return _ROS2_AVAILABLE

    def start(self):
        if not _ROS2_AVAILABLE:
            raise RuntimeError(
                "ROS 2 tidak tersedia. Install ROS 2 Humble dan jalankan:\n"
                "    source /opt/ros/humble/setup.bash"
            )
        if not rclpy.ok():
            rclpy.init()

        self._perception_node = PerceptionSubscriberNode(self._frame_queue)
        ros_node = self._perception_node.get_ros_node()

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(ros_node)

        self._running = True
        self._thread = threading.Thread(
            target=self._spin, daemon=True, name='ros2_spin'
        )
        self._thread.start()
        print('[ROSBagRunner] Started. Waiting for bag playback...')
        print(f'[ROSBagRunner] Topics: {TOPIC_RGB}, {TOPIC_DEPTH}')

    def _spin(self):
        try:
            self._executor.spin()
        except Exception as e:
            if self._running:
                print(f'[ROSBagRunner] spin error: {e}')

    def get_frame(self, timeout: float = 0.1) -> Optional[dict]:
        """Blocking get dengan timeout. Return None jika kosong."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._running = False
        if self._executor:
            self._executor.shutdown()
            self._executor = None
        if rclpy.ok():
            rclpy.shutdown()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print('[ROSBagRunner] Stopped.')
