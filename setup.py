from pathlib import Path

from setuptools import find_packages, setup

meta = {}
ROOT = Path(__file__).parent.resolve()
exec((ROOT / "gobbler/meta.py").read_text(), meta)
NAME, VERSION, HOMEPAGE = meta["name"], meta["version"], meta["homepage"]
long_description = (ROOT / "README.md").read_text()


setup(
    name=NAME,
    version=VERSION,
    author="Zifo RnD",
    author_email="nitesh.y@zifornd.com",
    description="Multimodal data ingestion and retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=HOMEPAGE,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "python-dotenv",
        "transformers==4.40.0",
        "opencv-python==4.10.0.84",
        "scikit-image",
        "numpy",
        "pydantic",
        "openai==1.88.0",
        "pymupdf==1.26.3",
        "keybert==0.9.0",
        "requests",
        "appdirs==1.4.4",
        "pymilvus==2.6.0b0",
        "agentic-doc==0.3.1",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
    platforms="Posix; Windows",
    keywords=["multimodal"],
    python_requires=">=3.10",
    license="PROPRIETARY",
)
