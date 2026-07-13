import os

from setuptools import find_packages, setup

base_dir = os.path.dirname(os.path.abspath(__file__))


def get_long_description():
    with open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open(os.path.join(base_dir, "requirements.txt"), encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="voxa",
    version="0.1.0",
    license="MIT",
    description="Voxa — standalone self-hosted speech-to-text",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=get_requirements(),
    extras_require={
        "dev": ["pytest==7.*", "black==23.*", "flake8==6.*", "isort==5.*"],
    },
    packages=find_packages(include=["voxa", "voxa.*"]),
    include_package_data=True,
    package_data={"voxa.engine": ["assets/*"], "voxa": ["web/*"]},
    entry_points={"console_scripts": ["voxa = voxa.server:main"]},
)
