from setuptools import setup, find_packages

setup(
    name = 'VisiTranscrible AI',
    Version = '1.0',
    packages = find_packages(where='src'),
    package_dir = {"": "src"},
    install_requires = [
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
        "pyyaml",
        "pillow",
        "python-multipart",
        "pydub",
        "openai-whisper"
    ],
)
