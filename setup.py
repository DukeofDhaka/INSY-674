from pathlib import Path

from setuptools import find_packages, setup

NAME = "insy674-end-to-end-ml"
DESCRIPTION = "End-to-end ML project with training pipeline and FastAPI serving"
URL = "https://github.com/tyagi14/INSY-674"
AUTHOR = "Akshit Tyagi"
REQUIRES_PYTHON = ">=3.10"

ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "src"

about = {}
about["__version__"] = (PACKAGE_DIR / "VERSION").read_text(encoding="utf-8").strip()


def list_reqs(fname: str = "production.txt") -> list[str]:
    reqs = (REQUIREMENTS_DIR / fname).read_text(encoding="utf-8").splitlines()
    reqs = [line.strip() for line in reqs if line.strip() and not line.startswith("#")]
    collected: list[str] = []
    for req in reqs:
        if req.startswith("-r "):
            nested_file = req.split(" ", maxsplit=1)[1]
            collected.extend(list_reqs(nested_file))
        else:
            collected.append(req)
    return collected


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"src": ["VERSION", "config.yml"]},
    install_requires=list_reqs(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
