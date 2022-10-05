import setuptools

setuptools.setup(
    name="utils3d",
    version="0.0.1",
    author="EasternJournalist@github.com",
    author_email="wangrc2081cs@mail.ustc.edu.cn",
    description="A small package for 3D graphics",
    long_description="A small package for 3D graphics",
    long_description_content_type="text/markdown",
    url="https://github.com/EasternJournalist/utils3d",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "moderngl",
        "numpy",
    ]
)

