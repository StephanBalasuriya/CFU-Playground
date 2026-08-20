# MobileNetV2 vs MobileNetV3: Comprehensive Guide

This document combines the technical comparisons between **MobileNetV2** and **MobileNetV3**, as well as the detailed breakdown of the **MobileNetV3 variants** (Large, Small, and Minimalistic).

---

## Part 1: Differences Between MobileNetV2 and MobileNetV3

MobileNetV2 and MobileNetV3 are lightweight deep neural network architectures designed by Google for mobile and edge vision applications. MobileNetV3 builds directly upon the inverted residual structures introduced in MobileNetV2, incorporating Automated Machine Learning (AutoML) and several micro-architectural enhancements.

### Key Technical Differences

#### 1. Architecture Design Methodology
* **MobileNetV2**: Designed manually using human intuition. It introduced **Inverted Residual Blocks** with **Linear Bottlenecks** (expanding channel counts before depthwise convolution and projecting back with a linear activation).
* **MobileNetV3**: Designed using **Platform-Aware Neural Architecture Search (NAS)** combined with the **NetAdapt** algorithm to automatically find optimal layer configurations, channel counts, and block placements tailored for mobile hardware, then fine-tuned manually.

#### 2. Squeeze-and-Excitation (SE) Attention Modules
* **MobileNetV2**: Does not use attention mechanisms in its core building blocks.
* **MobileNetV3**: Integrates **Squeeze-and-Excitation (SE)** modules into the inverted residual blocks. The SE mechanism dynamically re-weights channel feature maps based on global context, significantly boosting accuracy with minimal latency penalty.

#### 3. Activation Functions (`h-swish` and `h-sigmoid`)
* **MobileNetV2**: Uses **ReLU6** (`min(max(0, x), 6)`) across all layers to prevent precision loss in low-bit integer quantized execution.
* **MobileNetV3**: Uses **Hard-Swish (`h-swish`)** and **Hard-Sigmoid (`h-sigmoid`)** in deeper layers. Non-linearities like standard Swish (`x * sigmoid(x)`) are effective but computationally expensive due to exponential operations. MobileNetV3 approximates them using piecewise linear functions:
  * `h-sigmoid(x) = ReLU6(x + 3) / 6`
  * `h-swish(x) = x * h-sigmoid(x) = x * (ReLU6(x + 3) / 6)`
  * *Note: Swish/h-swish is applied in deeper layers where feature maps are smaller, mitigating memory lookup overhead.*

#### 4. Redesigned Efficient Head and Tail
* **MobileNetV2**: The output head expands feature maps to 1280 channels *before* global average pooling, resulting in high latency in the final layers.
* **MobileNetV3**: Redesigned the network head by moving **Global Average Pooling** *before* the final 1x1 expansion convolution. This reduces computation and latency in the last stage by ~7% without accuracy loss. It also reduced the initial stem conv filters from 32 to 16.

#### 5. Model Variants
* **MobileNetV2**: Scaled using a single width multiplier parameter $\alpha$ (e.g., 1.0x, 0.75x, 0.5x).
* **MobileNetV3**: Comes in standard predefined configurations targeting different latency budgets (Large and Small).

### MobileNetV2 vs MobileNetV3 Summary Table

| Feature / Metric | MobileNetV2 | MobileNetV3 |
| :--- | :--- | :--- |
| **Design Approach** | Manual engineering | Hardware-Aware NAS + NetAdapt + Human Tuning |
| **Basic Building Block** | Inverted Residuals + Linear Bottlenecks | Inverted Residuals + SE Modules + Linear Bottlenecks |
| **Attention Mechanism** | None | Squeeze-and-Excitation (SE) in Bottlenecks |
| **Primary Activations** | ReLU6 | ReLU6 (early layers) + Hard-Swish (deeper layers) |
| **Network Head Design** | 1x1 Expansion -> Pooling -> Classifier | Pooling -> 1x1 Expansion -> Classifier (7% faster) |
| **Standard Variants** | Single baseline (scaled via width multiplier) | MobileNetV3-Large & MobileNetV3-Small |
| **Relative Performance** | Baseline reference | ~20% faster than V2 at equal accuracy (or ~3.2% more accurate at equal latency) |

---

## Part 2: Differences Between MobileNetV3 Variants (Large, Small, Minimalistic)

MobileNetV3 comes in three main flavor categories: **Large**, **Small**, and **Minimalistic**. While **Large** and **Small** differ primarily in model capacity and depth, **Minimalistic** variants are specifically designed for hardware acceleration and embedded deployment.

### 1. MobileNetV3-Large
* **Target**: High-performance mobile devices (smartphones, edge devices with CPU/GPU).
* **Architecture**:
  * 15 Inverted Residual bottleneck blocks.
  * Uses **Squeeze-and-Excitation (SE)** attention modules in selected bottleneck layers.
  * Uses **`h-swish`** non-linear activation functions in middle and deeper layers.
  * Larger channel widths and layer depth.
* **Performance**: ~75.2% Top-1 ImageNet accuracy (~5.4 million parameters, ~219 MAdds).

### 2. MobileNetV3-Small
* **Target**: Resource-constrained mobile devices or latency-critical applications.
* **Architecture**:
  * 11 Inverted Residual bottleneck blocks (fewer layers and reduced channel dimensions).
  * Uses **SE modules in nearly all blocks** to compensate for the smaller network capacity.
  * Uses **`h-swish`** activations in deeper layers.
* **Performance**: ~67.4% Top-1 ImageNet accuracy (~2.5 million parameters, ~66 MAdds).

### 3. MobileNetV3-Minimalistic (Large & Small)
* **Target**: Low-end hardware accelerators, Edge NPUs/TPUs, DSPs, microcontrollers, and INT8 quantized hardware pipelines.
* **Key Structural Differences**:
  1. **No Squeeze-and-Excitation (SE) Modules**: Completely removes all SE attention blocks. SE modules require global pooling and two fully-connected layers inside convolution blocks, which are difficult to map efficiently to dedicated hardware accelerators.
  2. **No `h-swish` or `h-sigmoid` Activations**: Replaces all `h-swish` activations throughout the entire network back with standard **ReLU6**.
  3. **Preserves Network Topology**: Keeps the same layer counts, kernel sizes (3x3 depthwise), and channel expansion ratios discovered by NAS for the Large or Small variants.

### MobileNetV3 Variants Comparison Table

| Feature / Variant | MobileNetV3-Large | MobileNetV3-Small | MobileNetV3-Minimalistic |
| :--- | :--- | :--- | :--- |
| **Primary Goal** | Maximize accuracy for mobile CPUs | Minimal latency for mobile CPUs | Maximum compatibility for hardware accelerators / MCUs |
| **Layer Depth (Bottlenecks)** | 15 blocks | 11 blocks | Matches Large or Small (15 or 11 blocks) |
| **Squeeze-and-Excitation (SE)** | Yes (Selected blocks) | Yes (Almost all blocks) | **None** (Removed) |
| **Activation Functions** | ReLU + `h-swish` | ReLU + `h-swish` | **ReLU6 only** |
| **Hardware Compatibility** | Mobile CPU / GPU | Mobile CPU / GPU | **DSP, Edge TPU, NPU, Microcontrollers (TFLite Micro)** |
| **Quantization Readiness** | Standard | Standard | **Optimal for 8-bit Integer Quantization (INT8)** |

---

### Embedded & Hardware Accelerator Considerations

Standard MobileNetV3 features like **SE layers** (which require global average pooling in the middle of conv blocks) and **`h-swish`** (piecewise divisions) can be very slow or unsupported on microcontrollers, hardware accelerators, and custom FPGA engines (such as CFU-Playground execution targets).

**MobileNetV3-Minimalistic** acts as a bridge between MobileNetV2 and MobileNetV3: it benefits from the optimized layer dimensions found by Neural Architecture Search (NAS), but sticks strictly to simple **3x3 Depthwise Conv + 1x1 Pointwise Conv + ReLU6**, making it ideal for deployment on microcontrollers, custom hardware accelerators, and INT8 quantized engines.
