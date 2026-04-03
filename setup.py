from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="chronohorn",
    version="0.1.0",
    description="Runtime, replay, export, and frontier execution surface for predictive-coder descendants.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26",
    ],
    extras_require={
        "train": ["sentencepiece>=0.2"],
        "torch": ["sentencepiece>=0.2", "torch>=2.8"],
        "metal": ["sentencepiece>=0.2", "mlx>=0.25"],
    },
    entry_points={
        "console_scripts": [
            "chronohorn=chronohorn.cli:main",
            "chronohorn-mcp=chronohorn.mcp_transport:main",
        ]
    },
)
