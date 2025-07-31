# Installation Documentation

**Repository**: neuralmagic/speculators
**Generated with**: LLM-powered analysis
**Standards**: dita
**Focus**: Repository-specific installation documentation

---

# Installation Guide for neuralmagic/speculators

This document provides comprehensive instructions for installing `neuralmagic/speculators`, a powerful library designed for machine learning development, particularly focusing on efficient model inference and optimization. Whether you intend to use `speculators` as a dependency in your project or contribute to its development, this guide covers all necessary steps for a successful setup.

## 1. Introduction to neuralmagic/speculators

`neuralmagic/speculators` is an advanced Python library engineered to enhance the performance and capabilities of machine learning models, especially within the context of large language models (LLMs) and deep learning frameworks. It integrates seamlessly with cutting-edge technologies like vLLM, PyTorch, and Hugging Face Transformers, providing utilities for model configuration, data handling, and robust testing. Its design emphasizes efficiency and scalability, making it an invaluable tool for developers working on high-performance ML applications.

This guide will walk you through the process of setting up `neuralmagic/speculators` on your system, covering both standard package installation and installation directly from the source repository for development purposes.

## 2. Prerequisites

Before proceeding with the installation of `neuralmagic/speculators`, ensure your system meets the following requirements:

*   **Operating System**: Linux (Ubuntu, CentOS, etc.) or macOS are the recommended operating systems. While `speculators` may function on Windows, full compatibility and optimal performance, especially with GPU-accelerated components like vLLM and DeepSpeed, are best achieved on Unix-like environments.
*   **Python**: `neuralmagic/speculators` requires Python 3.8 or newer. It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with system-wide Python packages.
    *   You can check your Python version using:
        ```bash
        python3 --version
        ```
*   **Git**: Git is essential for cloning the `neuralmagic/speculators` repository if you plan to install from source or contribute to the project.
    *   Install Git if you haven't already:
        ```bash
        # On Debian/Ubuntu
        sudo apt update && sudo apt install git

        # On macOS (with Homebrew)
        brew install git
        ```
*   **Virtual Environment Tool**: Using a virtual environment (like `venv` or `conda`) is crucial for isolating project dependencies.
    *   **`venv` (recommended for most users)**: Built-in with Python 3.
    *   **`conda` (for Anaconda/Miniconda users)**: Useful for managing complex environments, especially those involving specific CUDA versions or scientific computing packages.
*   **GPU Hardware and Drivers (Optional but Recommended for ML Acceleration)**:
    *   If you intend to leverage GPU acceleration with frameworks like PyTorch, vLLM, or DeepSpeed, an NVIDIA GPU with CUDA support is highly recommended.
    *   Ensure you have the appropriate NVIDIA drivers and CUDA Toolkit installed. Refer to the NVIDIA developer website for the latest installation instructions specific to your GPU and operating system. `speculators` itself does not directly require CUDA, but its core dependencies for high-performance ML inference often do.

## 3. Installation Methods

There are two primary ways to install `neuralmagic/speculators`: via the Python Package Index (PyPI) for general use, or directly from the source repository for development and contribution.

### 3.1. Method 1: Installing from PyPI (Standard Package Installation)

This is the simplest method for users who want to integrate `neuralmagic/speculators` into their existing projects without modifying its source code.

1.  **Create and Activate a Virtual Environment**:
    It is strongly advised to create a dedicated virtual environment for `speculators` to manage its dependencies effectively.

    ```bash
    # Create a virtual environment named 'speculators_env'
    python3 -m venv speculators_env

    # Activate the virtual environment
    source speculators_env/bin/activate
    ```

    (For `conda` users):
    ```bash
    conda create -n speculators_env python=3.9 # Or your preferred Python version
    conda activate speculators_env
    ```

2.  **Install `neuralmagic/speculators` via pip**:
    Once your virtual environment is active, you can install the latest stable version of `speculators` from PyPI:

    ```bash
    pip install speculators
    ```

    **Note on Optional Dependencies**: `neuralmagic/speculators` is designed to be modular and may offer optional dependencies for specific functionalities (e.g., integration with vLLM, DeepSpeed, or specific model types). If these are available, you might install them using "extras":

    ```bash
    # Example: Installing with vLLM and DeepSpeed support (if available as extras)
    pip install speculators[vllm,deepspeed]
    ```
    Please consult the official `setup.py` or project documentation for the exact names of available extras.

### 3.2. Method 2: Installing from Source (Recommended for Development)

Installing from source is ideal for developers who wish to contribute to `neuralmagic/speculators`, modify its core functionalities, or work with the latest unreleased features. This method involves cloning the repository and installing it in "editable" mode.

1.  **Clone the Repository**:
    First, clone the `neuralmagic/speculators` repository from GitHub:

    ```bash
    git clone https://github.com/neuralmagic/speculators.git
    ```

2.  **Navigate into the Project Directory**:
    Change your current directory to the newly cloned `speculators` repository:

    ```bash
    cd speculators
    ```

3.  **Create and Activate a Virtual Environment**:
    As with PyPI installation, create and activate a virtual environment within the project directory. This ensures that all dependencies are isolated to this project.

    ```bash
    # Create a virtual environment
    python3 -m venv venv

    # Activate the virtual environment
    source venv/bin/activate
    ```

4.  **Install `neuralmagic/speculators` in Editable Mode**:
    Install the project in editable mode. This allows you to make changes to the source code, and those changes will be immediately reflected in your environment without needing to reinstall. The `setup.py` file handles the package definition and dependency resolution.

    ```bash
    pip install -e .
    ```
    This command will install `speculators` and all its core dependencies listed in `setup.py`. The `setup.py` file, which includes functions like `get_last_version_diff`, `get_next_version`, and `write_version_files`, is crucial for managing the project's versioning and packaging.

5.  **Install Core ML Framework Dependencies**:
    `neuralmagic/speculators` heavily relies on and integrates with various ML frameworks. You will need to install these separately, often with specific configurations for GPU support.

    *   **PyTorch**: Install PyTorch with CUDA support for GPU acceleration. Visit the official PyTorch website (pytorch.org) for the exact installation command tailored to your CUDA version and operating system. An example for CUDA 11.8:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   **vLLM**: A high-throughput inference engine for LLMs. vLLM requires specific CUDA versions and is highly optimized for NVIDIA GPUs. Refer to vLLM's official documentation for installation instructions.
        ```bash
        pip install vllm
        ```
        Ensure your CUDA setup is compatible with vLLM's requirements.
    *   **DeepSpeed**: A deep learning optimization library. DeepSpeed installation can be complex due to its MPI and CUDA dependencies. Consult the DeepSpeed GitHub repository for detailed instructions.
        ```bash
        pip install deepspeed
        ```
    *   **Hugging Face Transformers**: A widely used library for pre-trained models.
        ```bash
        pip install transformers
        ```
    *   **Other Models/Frameworks**: `speculators` also interacts with models like Llama, Mistral, and potentially frameworks like EAGLE. These might require additional model weights or specific configurations, which are typically handled within the `speculators` library's usage rather than during its installation.

6.  **Install Development Dependencies (Optional but Recommended for Contributors)**:
    If you plan to contribute to `neuralmagic/speculators`, you should install the development tools used for linting, formatting, and testing. These are typically listed in a `requirements-dev.txt` file or as `[dev]` extras in `setup.py`.

    ```bash
    # If a requirements-dev.txt exists
    pip install -r requirements-dev.txt

    # Alternatively, if defined as an extra in setup.py
    pip install -e .[dev]
    ```
    These dependencies include tools like `Black` (code formatter), `Mypy` (static type checker), `Pytest` (testing framework), `Ruff` (linter), and `Tox` (automation for testing in multiple environments).

## 4. Verifying the Installation

After completing the installation, it's crucial to verify that `neuralmagic/speculators` and its dependencies are correctly set up.

1.  **Basic Python Import Test**:
    Open a Python interpreter within your activated virtual environment and try importing the `speculators` package:

    ```bash
    python
    >>> import speculators
    >>> print(speculators.__version__) # If __version__ is defined
    >>> exit()
    ```
    If no errors occur, the basic package installation is successful.

2.  **Run Unit Tests**:
    `neuralmagic/speculators` includes a comprehensive suite of unit tests to ensure functionality. Running these tests is the most robust way to verify your installation, especially for development setups. The `tests/unit/test_model.py` file, for instance, contains tests for model functionalities like `reload_and_populate_configs` and `reload_and_populate_models`.

    Navigate to the root of the `speculators` repository (if you installed from source) and run `pytest`:

    ```bash
    # Ensure pytest is installed (it should be if you installed dev dependencies)
    pip install pytest

    # Run all tests
    pytest

    # To run specific tests, e.g., from test_model.py
    pytest tests/unit/test_model.py
    ```
    All tests should pass. Any failures indicate a potential issue with dependencies or the installation process.

3.  **Check GPU Availability (if applicable)**:
    If you installed PyTorch with CUDA support, verify that your GPU is recognized:

    ```bash
    python
    >>> import torch
    >>> print(torch.cuda.is_available())
    >>> print(torch.cuda.device_count())
    >>> print(torch.cuda.get_device_name(0))