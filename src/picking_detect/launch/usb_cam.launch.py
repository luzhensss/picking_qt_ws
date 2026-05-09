from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[
                {'video_device': '/dev/video0'},  # 摄像头设备路径
                {'pixel_format': 'yuyv'},       # 图像格式（mjpeg 或 yuyv）
                {'image_width': 1280},           # 宽度
                {'image_height': 720},           # 高度
                {'framerate': 30.0},               # 帧率
            ],
            output='screen'  # 打印日志到终端
        )
    ])

