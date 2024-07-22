from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'zed_streaming'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.xml')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='developer@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zed_streaming_rgbd = zed_streaming.zed_streaming_rgbd:main',
            'zed_streaming_rgb = zed_streaming.zed_stream_ros2_rgb:main',
            'zed_streaming_rgb_pcd = zed_streaming.zed_streaming_rgb_pcd:main',
        ],
    },
)
