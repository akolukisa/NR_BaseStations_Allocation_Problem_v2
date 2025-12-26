# 📡 5G RIC Beam Assignment Dashboard

Interactive web-based dashboard for visualizing and analyzing beam-user assignment optimization in 5G networks.

## Features

✅ **Real-time Visualization**
- Network topology with gNB and UE positions
- Beam-UE assignment mapping
- SINR heatmap
- Beam load distribution

✅ **Algorithm Comparison**
- GA (Genetic Algorithm)
- HGA (Hybrid Genetic Algorithm)
- PBIG (Population-Based Iterated Greedy)
- Max-SINR (Baseline)

✅ **Interactive Controls**
- UE count selection (4-100)
- Beam count selection (4-16)
- Alpha parameter tuning (Sum-Rate vs Fairness trade-off)
- Beam capacity limiting
- Interference factor adjustment

✅ **Real Data Support**
- Load actual 5G-LENA SINR measurements
- Or generate synthetic scenarios

## Installation

```bash
pip install streamlit plotly numpy pandas matplotlib
```

## Usage

### Start the Dashboard

```bash
cd /Users/akolukisa/FinalThesis/ric-python
streamlit run dashboard_ric.py
```

The app will open at `http://localhost:8501`

### How to Use

1. **Configure Simulation** (Left sidebar):
   - Select scenario type (Real 5G-LENA or Interactive Test)
   - Set number of UEs and Beams
   - Select algorithms to compare
   - Adjust parameters (α, interference factor, etc.)

2. **Run Simulation**:
   - Click "▶️ Run Simulation" button
   - Wait for algorithms to compute optimal assignments

3. **View Results**:
   - **Left Panel**: Network topology with beam assignments
   - **Right Panel**: Algorithm performance metrics table
   - **Bottom**: SINR heatmap and beam utilization

## Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│                    📡 Dashboard Header                   │
├──────────────────┬──────────────────────────────────────┤
│   ⚙️ SIDEBAR     │                                       │
│  Controls:       │    🗺️ NETWORK VISUALIZATION          │
│  - Scenario      │    (gNB positions, UE assignments)   │
│  - UE/Beam count │                                       │
│  - Algorithms    │                                       │
│  - Parameters    │                                       │
│  - Buttons       │                                       │
├──────────────────┼──────────────────────────────────────┤
│                  │    📊 RESULTS TABLE                   │
│                  │    (Algorithm comparison)             │
├──────────────────┴──────────────────────────────────────┤
│        📈 DETAILED ANALYSIS (Bottom)                     │
│  ┌─────────────────┐  ┌──────────────────┐              │
│  │  SINR Heatmap   │  │ Beam Distribution │              │
│  └─────────────────┘  └──────────────────┘              │
└───────────────────────────────────────────────────────┘
```

## Parameters Explained

### α (Alpha)
- **0.0**: Prioritize fairness (Jain index)
- **0.5**: Balanced throughput & fairness
- **1.0**: Maximize total throughput

### Beam Capacity Limit
- Max UEs assigned to same beam
- Prevents over-utilization
- Default: 4 UEs/beam (3GPP typical)

### Interference Factor
- Penalty applied per additional UE on same beam
- Formula: `SINR_eff = SINR - factor × (num_ues - 1)`
- Typical: 0.5 dB/UE

## Data Sources

### Real 5G-LENA Data
- **Path**: `lena_scalability_8beam/ue{10,15,20,...,50}/`
- **Files**: `sinr_gnb{0,1,2}_{200ms,300ms}.json`
- **Format**: JSON with SINR matrix per gNB

### Synthetic Data
- Randomly generated SINR values (5-25 dB range)
- Used for quick testing and demos

## Output Metrics

| Metric | Description |
|--------|-------------|
| **Sum-Rate** | Total throughput (bps/Hz) = Σ log₂(1 + SINR) |
| **Throughput** | Practical Mbps = Sum-Rate × 100 MHz × 0.8 |
| **Jain Index** | Fairness measure (0=unfair, 1=perfect fair) |
| **Fitness** | Combined metric: F = α×SR + (1-α)×Jain |
| **Runtime** | Algorithm execution time in milliseconds |

## Performance Benchmarks (8 Beam, 30 UEs)

| Algorithm | Sum-Rate | Throughput | Jain | Runtime |
|-----------|----------|-----------|------|---------|
| HGA ⭐   | 194.47 | 15,558 Mbps | 0.4215 | 1.48s |
| PBIG | 187.88 | 15,030 Mbps | 0.4072 | 0.44s |
| GA | 179.19 | 14,335 Mbps | 0.3763 | 0.21s |
| Max-SINR | 169.65 | 13,572 Mbps | 0.3958 | 0.00s |

## Troubleshooting

### Streamlit Not Found
```bash
pip install --upgrade streamlit
```

### Port Already in Use
```bash
streamlit run dashboard_ric.py --server.port 8502
```

### SINR Data Not Loading
- Verify `lena_scalability_8beam/` directory exists
- Check file format is correct JSON
- Falls back to synthetic data automatically

## Future Enhancements

- [ ] Real-time algorithm execution with progress bar
- [ ] Multi-gNB coordination visualization
- [ ] Beam pattern visualization (antenna patterns)
- [ ] Channel state information (CSI) uploads
- [ ] Algorithm parameter tuning interface
- [ ] Export results to CSV/PDF
- [ ] Comparison with ns-3 simulations

## References

- 5G-LENA: https://5g-lena.cttc.es/
- O-RAN: https://www.o-ran.org/
- Streamlit: https://streamlit.io/

---
**Dashboard Version**: 1.0  
**Created**: 2025-12-25  
**Author**: RIC Optimization Team
