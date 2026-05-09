from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'picking_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
        
        # # 包含 detect 目录及其所有子目录
        # (os.path.join('share', package_name, 'detect'), glob('detect/*.py')),
        # (os.path.join('share', package_name, 'detect/core'), glob('detect/core/*.py')),
        # (os.path.join('share', package_name, 'detect/utils'), glob('detect/utils/*.py')),
        # (os.path.join('share', package_name, 'detect/config'), glob('detect/config/*.py')),
        # (os.path.join('share', package_name, 'detect/models'), glob('detect/models/*.py')),
        # (os.path.join('share', package_name, 'detect/dtos'), glob('detect/dtos/*.py')),

        # 包含 detect_trt 目录及其所有子目录
        (os.path.join('share', package_name, 'detect_trt'), glob('detect_trt/*.py')),
        (os.path.join('share', package_name, 'detect_trt/core'), glob('detect_trt/core/*.py')),
        (os.path.join('share', package_name, 'detect_trt/utils'), glob('detect_trt/utils/*.py')),
        (os.path.join('share', package_name, 'detect_trt/config'), glob('detect_trt/config/*.py')),
        (os.path.join('share', package_name, 'detect_trt/models'), glob('detect_trt/models/*.py')),
        (os.path.join('share', package_name, 'detect_trt/dtos'), glob('detect_trt/dtos/*.py')),
        (os.path.join('share', package_name, 'detect_trt/common'), glob('detect_trt/common/*.py')),
    ],
    
    install_requires=[
        'setuptools',
        # 'opencv-python>=4.5.0',
        # 'ultralytics>=8.0.0',
        # 'numpy>=1.21.0',
        # 'torch>=1.10.0',
        # 'torchvision>=0.11.0',
    ],
    zip_safe=False,
    maintainer='luzhens',
    maintainer_email='test@163.com',
    description='YOLOv8 object detection for robotic picking application',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'picking_detector = picking_detect.picking_detector:main',  
            'picking_test_detector = picking_detect.picking_test_detector:main',  
            'picking_detector_prod = picking_detect.picking_detector_prod:main',  
            'picking_detector_onnx = picking_detect.picking_detector_onnx:main',
            'picking_detector_onnx_ai = picking_detect.picking_detector_onnx_ai:main',
            'calibrate_hand_eye = picking_detect.calibrate_hand_eye:main',
            'calibrate_hand_eye_single = picking_detect.calibrate_hand_eye_single:main',
        ],
    },
)
