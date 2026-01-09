# EEfficient Hierarchical Scheduler for Deep Learning Clusters




This project implements a **hierarchical scheduling system** designed to maximize the computational efficiency of deep learning clusters. It operates on two complementary levels:

*   **Cluster-Level (ExplSched)**: At the cluster level, our scheduler `ExplSched` dynamically allocates cluster resources (e.g., GPUs) among competing exploratory training jobs (like hyperparameter tuning). Instead of naively minimizing job completion time, it intelligently prioritizes jobs that deliver higher **training utility per resource consumed**, significantly reducing resource waste from early-terminated jobs.

*   **Device-Level (Kernel Scheduler)**: Within a single device, fine-grained **kernel scheduling** is performed by the **Kernel Manager)** framework. It intercepts CUDA kernel calls (e.g., from cuBLAS/cuDNN) and records their metadata. Our **Interference-Aware Scheduling Algorithm** then groups these kernels for concurrent execution, strictly ensuring that the combined utilization of **SMs** and **memory bandwidth** does not exceed hardware limits. This prevents performance degradation from resource contention and ensures stable, efficient kernel execution.


[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![C++ Version](https://img.shields.io/badge/C++-17%2B-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.6-yellow.svg)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://elasticmm.readthedocs.io/)




## 🚀 Key Features

- **Hierarchical Co-Scheduling:** The system achieves full-stack resource optimization from jobs down to operators through the co-design of the cluster-level utility scheduler (ExplSched) and the device-level kernel scheduler.
- **Smart Scheduling for Exploratory Jobs:** The cluster scheduler intelligently allocates resources with the goal of maximizing training utility, significantly reducing GPU resource waste caused by the early termination of experimental tasks.
- **Interference-Aware Fine-Grained Execution:** The device-level scheduler performs fine-grained control and orchestration of concurrently executing kernels on the GPU, strictly avoiding resource contention to ensure stable and efficient high-performance computing.


## 🛠️ Technology Stack
Core Languages: C++ (for high-performance scheduler and operator interception), Python (for high-level APIs and job submission).

GPU Computing & Scheduling: CUDA Runtime/Driver API, cuBLAS, cuDNN (for kernel interception and execution).

System Libraries: dlfcn (dlopen, dlsym for dynamic symbol loading and API redirection).

Cluster Management (Optional Integration): Slurm, Kubernetes (for multi-node job deployment).

Build Tools: CMake, GCC/Clang.



Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **NCCL**: For multi-GPU communication (usually included with PyTorch)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/CapitalLiu/ElasticMM.git
cd ElasticMM

# Install the package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## 🚀 Quick Start

### Step 1: System Calibration (Recommended)

Before running inference, we recommend calibrating the system parameters for your hardware configuration. This offline profiling step helps optimize performance for your specific GPU setup.

```bash
# Run calibration to profile your machine's parameters
python examples/calibrate_gain_cost.py
```

The calibration process will:
- Profile GPU memory and compute capabilities
- Measure KV cache transfer bandwidth
- Calculate optimal resource allocation parameters
- Generate configuration files for your hardware

**Note**: This step is optional but highly recommended for optimal performance.

### Step 2: Simple Usage Example

For a basic example demonstrating ElasticMM's core functionality:

```bash
python examples/simple_usage.py
```

This example shows:
- Basic system initialization
- Request submission and processing
- Output collection and handling

### Step 3: Online Service with Dynamic Workload

For a complete online service demonstration with dynamic request generation:

```bash
python examples/demo_with_workload.py
```

This demo includes:
- Full system deployment with proxy and scheduler
- Dynamic request generation (text-only and multimodal)
- Real-time load balancing and auto-scaling
- Performance monitoring and statistics

### System Requirements

⚠️ **Important**: ElasticMM requires a minimum of **8 GPUs** to run with the default configuration (2 for text-only workloads + 6 for multimodal workloads).

- **Minimum**: 8 GPUs
- **Recommended**: 8+ GPUs with high-bandwidth interconnects (NVLink/InfiniBand) for optimal performance
- **Memory**: Sufficient GPU memory for your target model (typically 20GB+ per GPU)

## 🏗️ Architecture

ElasticMM implements a hierarchical architecture with two main levels:

1. **Modality Level**: Allocates GPU instances between text-only and multimodal workloads
2. **Stage Level**: Manages encoding, prefill, and decoding stages within each modality group

The system automatically balances resources based on real-time demand and performance metrics, ensuring optimal utilization across different workload types.

## 📊 Performance

- **High Throughput**: Optimized for maximum requests per second
- **Low Latency**: Minimized time-to-first-token (TTFT)
- **Efficient Resource Usage**: Dynamic allocation prevents resource waste
- **Scalable**: Supports from single GPU to multi-node deployments


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [vLLM](https://github.com/vllm-project/vllm) for efficient LLM serving
- Inspired by research in elastic computing and multimodal systems
- Thanks to the open-source community for various dependencies

## 📚 Citation

If you find ElasticMM useful in your research or production deployments, please cite our NeurIPS 2025 paper:

```
@inproceedings{liu2025elasticmm,
  title     = {ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism},
  author    = {Liu, Zedong and Cheng, Shenggan and Tan, Guangming and You, Yang and Tao, Dingwen},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.10069}
}
```


**ElasticMM** - Making multimodal AI more efficient and accessible.
