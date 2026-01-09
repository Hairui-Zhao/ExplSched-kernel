# Efficient Hierarchical Scheduler for Deep Learning Clusters




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
**Core Languages:** C++ (for high-performance scheduler and operator interception), Python (for high-level APIs and job submission).

**GPU Computing & Scheduling:** CUDA Runtime/Driver API, cuBLAS, cuDNN (for kernel interception and execution).

**System Libraries:** dlfcn (dlopen, dlsym for dynamic symbol loading and API redirection).

**Cluster Management:** Slurm, Kubernetes (for multi-node job deployment).

**Build Tools:** CMake, GCC/Clang.


## 📥 Installation & Compilation
Below are the brief steps to compile and install the project from source on a Linux system.

### Clone Repository & Prepare Environment

```bash
git clone https://github.com/Hairui-Zhao/ExplSched.git
cd explsched
```

### Install Dependencies & Build

```bash
# Install essential system libraries like dlfcn (usually pre-installed).
sudo apt-get update && sudo apt-get install -y build-essential cmake
# Build the project using CMake.
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Configure & Run

```
bash
# 1. Set the dynamic library path to ensure the system can find the WFM interception library.
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
# 2. Inject the WFM library via preloading to intercept CUDA calls.
export LD_PRELOAD=$(pwd)/lib/libwfm_intercept.so
# 3. Start the cluster scheduler daemon.
./sbin/explsched_daemon --config ../conf/cluster_config.yaml
# 4. Submit a sample job using the Python client.
python ../examples/submit_exploratory_job.py
```


## 📄 License


ArrayPipe is released under the **MIT License**.  Please see the `LICENSE` file for full details.



## 📚 Citation

```
@inproceedings{explsched2023,
    title={ExplSched: Maximizing Deep Learning Cluster Efficiency for Exploratory Jobs},
    author={Li, Hongliang and Zhao, Hairui and Xu, Zhewen and Li, Xiang and Xu, Haixiao},
    booktitle={2023 IEEE International Conference on Cluster Computing (CLUSTER)},
    year={2023},
    pages={1-12},
    doi={10.1109/CLUSTER52292.2023.00014}
}
```

