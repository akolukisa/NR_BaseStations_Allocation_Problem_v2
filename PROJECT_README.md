# 5G-LENA Near-RT RIC Beam/UE Assignment Research Project

## Project Overview

This project implements an O-RAN-inspired Near-RT RIC (RAN Intelligent Controller) for beam and UE assignment optimization in 5G networks, integrated with the 5G-LENA NR module for ns-3.

### Research Goals

Compare beam/UE assignment performance between:

**Baseline Algorithms** (5G-LENA built-in):
- Max-SINR (Upper Bound)
- Exhaustive Search

**RIC-based Algorithms** (Python Near-RT RIC):
- GA (Genetic Algorithm)
- HGA (Hybrid Genetic Algorithm)
- PBIG (Priority-Based Iterative Greedy)

## Directory Structure

```
FinalThesis/
├── ns-3-dev/                      # ns-3 + 5G-LENA simulator
│   └── contrib/nr/               # 5G-LENA NR module
│       ├── model/                # Core NR models
│       ├── helper/               # Helper classes
│       └── examples/             # Example scenarios
├── ric-python/                   # Near-RT RIC (Python)
│   ├── ric_server.py            # Main RIC server with algorithms
│   └── test_ric_client.py       # Test client
├── configs/                      # Simulation configurations
└── analysis/                     # Results analysis scripts
```

## Setup Instructions

### 1. Prerequisites (macOS)

```bash
# Xcode command line tools
xcode-select --install

# Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake gcc python@3.11 gsl eigen

# Python packages
python3 -m pip install numpy scipy matplotlib pandas
```

### 2. Build ns-3 + 5G-LENA

The simulator has already been configured and built at:
```
/Users/akolukisa/FinalThesis/ns-3-dev
```

To rebuild:
```bash
cd /Users/akolukisa/FinalThesis/ns-3-dev
/opt/homebrew/bin/python3.11 ./ns3 configure --enable-examples --enable-tests
/opt/homebrew/bin/python3.11 ./ns3 build
```

To run an example:
```bash
cd /Users/akolukisa/FinalThesis/ns-3-dev
/opt/homebrew/bin/python3.11 ./ns3 run cttc-nr-demo
```

### 3. Test Python RIC Server

#### Start the RIC Server

```bash
cd /Users/akolukisa/FinalThesis/ric-python
chmod +x ric_server.py test_ric_client.py

# Start with GA algorithm
python3 ric_server.py --algorithm GA --port 5555

# Or try other algorithms:
# python3 ric_server.py --algorithm Max-SINR
# python3 ric_server.py --algorithm HGA
# python3 ric_server.py --algorithm PBIG
```

#### Test with the client (in another terminal)

```bash
cd /Users/akolukisa/FinalThesis/ric-python

# Test with default settings (8 beams, 4 UEs)
python3 test_ric_client.py --algorithm GA

# Test with custom scenario
python3 test_ric_client.py --algorithm HGA --beams 16 --ues 10
```

## Algorithm Details

### Baseline Algorithms (5G-LENA)

1. **Max-SINR**
   - Assigns each UE to the beam with highest SINR
   - Upper bound on performance (ignores interference/resource constraints)
   - Time complexity: O(B × U) where B=beams, U=UEs

2. **Exhaustive Search**
   - Tries all possible beam-UE assignments
   - Finds optimal solution
   - Time complexity: O(B^U) - only feasible for small scenarios

### RIC Algorithms (Python)

3. **GA (Genetic Algorithm)**
   - Population-based search
   - Operators: tournament selection, crossover, mutation
   - Parameters: population=50, generations=100, mutation_rate=0.1

4. **HGA (Hybrid Genetic Algorithm)**
   - GA + local search
   - Local search: greedy improvement per gene
   - Better exploitation vs pure GA

5. **PBIG (Priority-Based Iterative Greedy)**
   - Prioritizes UEs with less flexibility (low SINR variance)
   - Greedy beam selection per UE
   - Fast, scalable

## RIC Server API

### Request Format (ns-3 → RIC)

```json
{
  "scenario_id": 1,
  "gNB_id": 0,
  "num_beams": 8,
  "num_ues": 4,
  "sinr_matrix_dB": [[...], [...], ...]
}
```

`sinr_matrix_dB` is a 2D array: `[beam][ue]` containing SINR values in dB.

### Response Format (RIC → ns-3)

```json
{
  "algorithm": "GA",
  "beam_for_ue": [0, 2, 5, 1],
  "objective_value": 12.34,
  "scenario_id": 1
}
```

`beam_for_ue[i]` is the assigned beam index for UE `i`.

## Next Steps

### Integration with ns-3

To integrate the RIC with ns-3:

1. **Create RIC Client in C++**
   - Location: `ns-3-dev/contrib/nr/model/nr-ric-client.cc/h`
   - Functionality:
     - Collect SINR matrix from `NrGnbPhy`
     - Send JSON request to RIC server (TCP socket)
     - Parse response and apply beam assignments

2. **Hook into Scheduler**
   - Modify: `ns-3-dev/contrib/nr/model/nr-mac-scheduler-ofdma-symbol-per-beam.cc`
   - Add: `RicBeamAssignment` class that uses `NrRicClient`
   - Allow switching between baseline and RIC algorithms via attribute

3. **Example Scenario**
   - Create: `contrib/nr/examples/cttc-nr-ric-comparison.cc`
   - Compare all 5 algorithms on same channel realization
   - Output: throughput, fairness, cell-edge performance

### Experiment Design

**Scenarios to test:**
- UE count: 5, 10, 20, 40
- Beam count: 8, 16, 32
- Mobility: static, 3 km/h, 30 km/h
- Channel: LOS, NLOS, mixed

**Metrics:**
- Total cell throughput (Mbps)
- Average UE throughput
- Cell-edge throughput (5th percentile)
- Jain fairness index
- Algorithm runtime

## File Overview

### Python RIC (`ric-python/`)

- **`ric_server.py`**: Main RIC server
  - Classes: `RICServer`, `BeamAssignmentOptimizer`, `GAOptimizer`, `HGAOptimizer`, `PBIGOptimizer`
  - Features: TCP server, JSON API, multiple algorithms
  - Usage: `python3 ric_server.py --algorithm GA --port 5555`

- **`test_ric_client.py`**: Test client
  - Simulates ns-3 requests with synthetic SINR matrices
  - Validates RIC server responses

### ns-3 Examples (`ns-3-dev/contrib/nr/examples/`)

Current 5G-LENA examples:
- `cttc-nr-demo.cc`: Basic NR demo
- `cttc-nr-cc-bwp-demo.cc`: Component carriers and BWPs
- `cttc-realistic-beamforming.cc`: Realistic beamforming

To create (next phase):
- `cttc-nr-ric-comparison.cc`: Compare baseline vs RIC algorithms

## Key 5G-LENA Classes

### Beamforming

- **`BeamManager`** (`model/beam-manager.h`)
  - Manages beamforming vectors per device
  - Stores beam-device mappings

- **`IdealBeamformingAlgorithm`** (`model/ideal-beamforming-algorithm.h`)
  - Base class for beamforming algorithms
  - Method: `GetBeamformingVectors()`

### Scheduling

- **`NrMacScheduler`** (`model/nr-mac-scheduler.h`)
  - Base MAC scheduler

- **`NrMacSchedulerOfdmaSymbolPerBeam`** (`model/nr-mac-scheduler-ofdma-symbol-per-beam.h`)
  - Allocates symbols per beam
  - Variants: load-balanced, round-robin, proportional fair
  - **Hook point for RIC integration**

## Troubleshooting

### Python 3.14 incompatibility

If you see errors with `argparse`, use Python 3.11:
```bash
/opt/homebrew/bin/python3.11 ./ns3 configure --enable-examples --enable-tests
```

### Eigen3 not found

If Eigen3 is not detected:
```bash
brew install eigen
# Eigen3 is optional for MIMO features; basic NR works without it
```

### RIC Server connection refused

Make sure server is running:
```bash
# Terminal 1: Start server
python3 ric_server.py --algorithm GA

# Terminal 2: Test client
python3 test_ric_client.py
```

## Contact & Support

For questions about the research or implementation, refer to:
- 5G-LENA documentation: https://5g-lena.cttc.es/
- 5G-LENA user group: https://groups.google.com/g/5g-lena-users/
- ns-3 documentation: https://www.nsnam.org/documentation/

## References

- Dreifuerst & Heath (2023). "Baseline algorithms for beam/UE assignment"
- O-RAN Alliance specifications: https://www.o-ran.org/specifications
- 5G-LENA NR Module: https://gitlab.com/cttc-lena/nr
