from setuptools import find_packages, setup

package_name = 'adaptive_navigator'

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
    maintainer='gokulkishore',
    maintainer_email='gokulaenugu2005@gmail.com',
    description='a navigation package with adaptive search capabilities',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'publisher = adaptive_navigator.publisher:main',
            'subscriber = adaptive_navigator.subscriber:main',
        ],
    },
)
