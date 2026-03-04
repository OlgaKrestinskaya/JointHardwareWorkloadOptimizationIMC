# Joint Hardware-Workload Co-Optimization for In-Memory Computing Accelerators

This repository contains the official implementation of  
**Joint Hardware-Workload Co-Optimization Framework for In-Memory Computing Accelerators**.

> **Authors:** Olga Krestinskaya, Mohammed E. Fouda, Ahmed Eltawil, and Khaled N. Salama  
> Accepted to **IEEE Access**

---

## Overview

A joint hardware-workload co-optimization framework based on an optimized **evolutionary algorithm (EA)** for designing **generalized IMC accelerator architectures supporting multiple workloads**. 

### Key Features

- Joint optimization of **hardware parameters** and **workload characteristics**
- Supports optimization for **multiple neural network workloads** within a single accelerator design
- Reduced performance gap between **workload-specific** and **generalized** IMC architectures
- Compatible with **RRAM-based** and **SRAM-based IMC platforms**
- Robust and adaptable across diverse **design and technology scenarios**

The framework is built on top of:

- [**CiMLoop**](https://github.com/mit-emze/cimloop/tree/main)  

> _Illustrative figure to be added here._

- **Paper:** (link will be added)  
- **Citation:** (to be added)

---

## Repository Structure

```
HardwareWorkloadOptimization/
-- main/
-- --HWC_00_additionalLibraries.ipynb          # Load required libraries and helper utilities
-- --HWC_01_mainTest.ipynb                     # Main example demonstrating the Joint Hardware-Workload Co-Optimization framework
-- --HWC_02_testsForCOST.ipynb                 # Evaluation of cost metrics and architecture trade-offs
-- --HWC_03_SRAM9network_4phase.ipynb          # Four-phase evolutionary search for SRAM-based IMC architectures across 9 neural networks
-- --HWC_04_tryingAggregations.ipynb           # Experiments with multi-workload aggregation strategies (Part 1)
-- --HWC_04_tryingAggregations2.ipynb          # Experiments with multi-workload aggregation strategies (Part 2)
-- --HWC_05_dev_circ_arch_sys_search.ipynb     # Sequential exploration for RRAM-based IMC: device-circuit-architecture-system design space exploration
-- --HWC_05_dev_circ_arch_sys_searchSRAM.ipynb # Sequential exploration for SRAM-based IMC: circuit-architecture-system design space exploration
-- --HWC_06_separateGPT.ipynb                  # Example showing how to evaluate a single hardware configuration
-- models/                                     # Neural network models used for hardware-workload evaluation
```

---

## Requirements

- **Processor:** Multi-core CPU (64 cores used in our experiments recommended for faster search)

Required libraries are listed in:  
`HardwareWorkloadOptimization/main/testAndInstallations.ipynb`

---

## Installation and Initial Run

CIMNAS is built on [**CiMLoop**](https://github.com/mit-emze/cimloop/tree/main), which itself depends on [**Timeloop** and **Accelergy**](https://github.com/Accelergy-Project/timeloop-accelergy-exercises).  
These require Docker with `sudo` (admin) access.

Follow the guidelines in _docker-compose.yaml_ file in **Run as follows** section to set up USER_UID and USER_GID.

Follow steps similar to [CiMLoop](https://github.com/mit-emze/cimloop):

```bash
git clone https://github.com/OlgaKrestinskaya/JointHardwareWorkloadOptimizationIMC.git
cd HardwareWorkloadOptimization
export DOCKER_ARCH=<your processor architecture>  # e.g., amd64
docker-compose pull
docker-compose up
```

> **ARM64** is supported by Timeloop and Accelergy Docker,  
> but is marked *unstable* - building from source is recommended on ARM as per [CiMLoop](https://github.com/mit-emze/cimloop).

Access JupyterLab from your browser (unless port mapping is changed):  
`http://127.0.0.1:8888/lab`

---

## GPU Support

If you need to modify the Docker configuration to enable GPU support, please refer to the setup instructions provided in the [CIMNAS](https://github.com/OlgaKrestinskaya/CIMNAS) framework.


---

## Usage

- **Installations and debugging:**  
  `HardwareWorkloadOptimization/main/testAndInstallations.ipynb`
  
- **Run CIMNAS framework example:**  
  `HardwareWorkloadOptimization/main/HardwareWorkloadOptimization.ipynb`

- **Test a single architecture:**  
  `HardwareWorkloadOptimization/main/checking_singleHardwareArchitecture.ipynb`


---

## Acknowledgments


CIMNAS builds on:

- [CiMLoop](https://github.com/mit-emze/cimloop)

We thank the maintainers of these projects for their foundational work.

---
