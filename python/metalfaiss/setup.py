import setuptools
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(this_directory, "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metalfaiss",
    version="0.1.0",
    author="Sydney R.",
    author_email="sydney@solace.ofharmony.ai",
    description="A pure Python port of FAISS using MLX and Metal JIT acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sydneyrenee/MetalFaiss",
    package_dir={"": "metalfaiss"},
    packages=setuptools.find_packages(where="metalfaiss"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "mlx",
        "numpy",
    ],
)