from setuptools import find_packages, setup

package_name = 'extended_kalman_filter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amlab',
    maintainer_email='wldnjs946429@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_publisher = extended_kalman_filter.imuPublisher:main',
            'filter = extended_kalman_filter.EKF:main',
            'logger = extended_kalman_filter.dataLogger:main',
            
        ],
    },
)
