# Renode Setup and Execution Guide for CFU-Playground

This guide documents the complete step-by-step setup and troubleshooting instructions to run CFU-Playground projects in **Renode** simulation.

---

## 1. Initial Environment Setup

From the root directory of `CFU-Playground`, initialize and download all dependencies (Conda, GCC RISC-V toolchain, Yosys, Verilator, Python packages, submodules):

```bash
cd ~/cfu-playground-fork/CFU-Playground
make env
```

---

## 2. Renode CPU Configuration Fix

To ensure Renode properly models the VexRiscv processor with custom CSR instructions, configure `cpuType` in `scripts/generate_renode_scripts.py`:

In [scripts/generate_renode_scripts.py](file:///home/victus_linux/cfu-playground-fork/CFU-Playground/scripts/generate_renode_scripts.py), verify or add `cpuType: "rv32im_zicsr"` under `cpu:` in `generate_repl()`:

```python
cpu:
    cpuType: "rv32im_zicsr"
    init:
        RegisterCustomCSR "BPM" 0xB04  User
        RegisterCustomCSR "BPM" 0xB05  User
        ...
```

---

## 3. Toolchain & ISA Compatibility Notice

The conda environment provides **GCC 10.1.0** (`gcc-riscv32-elf-newlib`). 
- Do **not** hardcode `-march=rv32i2p0_m_zicsr` in `third_party/python/litex/litex/soc/software/common.mak`, as GCC 10.1.0 does not recognize the `_zicsr` march syntax (CSR support is implicit in `rv32im` in GCC 10).
- Keep `common.mak` with standard compiler invocations:
  ```makefile
  define compile
  $(CC) -c $(CFLAGS) $(1) $< -o $@
  endef

  define assemble
  $(CC) -c $(CFLAGS) -o $@ $<
  endef
  ```

---

## 4. Activating the Conda Environment

Before compiling or running any project, **always activate the conda environment** so that `yosys`, `riscv32-unknown-elf-gcc`, and `verilator` are in your `PATH`:

```bash
source ~/cfu-playground-fork/CFU-Playground/env/conda/bin/activate cfu-common
```
*(When activated, your prompt will show `(cfu-common)`).*

> **Note:** If you are inside a subdirectory like `proj/mnv2_first`, always use the full path `~/cfu-playground-fork/CFU-Playground/env/conda/bin/activate cfu-common` or `../../env/conda/bin/activate cfu-common`.

---

## 5. Building and Running a Project in Renode

1. Navigate to your target project folder (e.g. `mnv2_first`):
   ```bash
   cd ~/cfu-playground-fork/CFU-Playground/proj/mnv2_first
   ```

2. Clean any stale build artifacts:
   ```bash
   make clean
   ```

3. Build and launch the Renode simulation:
   ```bash
   make renode
   ```

---

## 6. Expected Output

When launched, the build will:
1. Generate the CFU Verilog using Amaranth + Yosys (`cfu_gen.py`).
2. Build LiteX software, libraries (`picolibc`), and project firmware (`software.bin`).
3. Compile the Verilator co-simulation library (`libVtop.so`).
4. Generate the Renode platform and script files (`digilent_arty.resc`, `digilent_arty.repl`).
5. Launch Renode interactively and display the CFU-Playground menu via UART:

```
Hello, World!

CFU Playground
==============
 1: TfLM Models menu
 2: Functional CFU Tests
 3: Project menu
 4: Performance Counter Tests
 5: TFLite Unit Tests
 6: Benchmarks
 7: Util Tests
 8: Embench IoT
main> 
```
