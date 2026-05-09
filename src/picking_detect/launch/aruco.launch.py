from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='aruco_ros',
            executable='single',
            name='aruco_single',
            remappings=[
                ('/camera_info', '/camera_info'),
                ('/image', '/image_raw'),
            ],
            parameters=[{
                'marker_size': 0.1,           # 你的Aruco码实际边长（米）
                'marker_id': 0,                # 要识别的Aruco码ID
                'camera_frame': 'camera_optical_frame',
                'marker_frame': 'aruco_marker',
                'reference_frame': 'camera_optical_frame',
            }]
        ),
    ])