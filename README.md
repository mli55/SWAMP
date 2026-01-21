# SWAMP

This repository provides a research code release for **Commodity Wireless Links Monitor Belowground Tuber Growth**  

SWAMP combines two complementary measurement modalities:

- **Channel Frequency Response (CFR)** sweeps collected with software-defined radios (SDRs)
- **LTE link and channel indicators** collected with an srsRAN-based private LTE setup

This public snapshot contains the components that can be shared at this time for data processing, feature extraction, simulation, and analysis. Some experiment-control and deployment code that is tightly coupled to lab-specific SDR hardware and local network configuration is not included.

---

## Features

- Non-invasive subsurface growth monitoring using wireless signals
- Hybrid sensing with CFR-derived spectral features and LTE-derived link metrics
- Reproducible analysis and figure-generation scripts
- Modular pipeline for preprocessing, feature extraction, modeling, and evaluation
- Simulation workspace for controlled experiments where applicable



## Repository Structure

Directory names may vary slightly depending on the snapshot:

```
SWAMP/
├─ cfr_gnuradio/        # CFR measurement scripts and processing utilities
├─ lte_srsran/          # srsRAN-based LTE stack and metric logging utilities
├─ simulation/          # Python package, scripts, and tests for simulation/analysis
└─ README.md            # This file
```

Some experiment orchestration and deployment components are intentionally omitted due to tight coupling with specific hardware setups.



## Requirements

- Python 3.8+ (3.10+ recommended)
- NumPy, SciPy
- matplotlib
- Optional: pandas, scikit-learn



## Installation

Clone the repository and install dependencies:

```bash
git clone <REPOSITORY_URL>.git
cd SWAMP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additional dependencies are managed under `simulation/`:

```bash
pip install -e ./simulation
```



## Data

Typical inputs include:
- CFR frequency sweeps (complex baseband responses)
- LTE time-series metrics (e.g., RSRP, SINR, MCS, BLER, throughput)

Refer to in-repository documentation for expected file formats and directory layouts.



## Usage

Entry points vary by snapshot. Start with:

- Analysis and simulation scripts in `simulation/`
- CFR processing utilities in `cfr_gnuradio/`
- LTE metric parsing in `lte_srsran/`

Most scripts support `--help` for usage instructions.

A typical workflow:
1. Prepare or convert datasets into the expected layout
2. Run preprocessing and feature extraction
3. Perform modeling and inference
4. Generate evaluation results and figures



## Citation

If you use this work, please cite the paper:

**Commodity Wireless Links Monitor Belowground Tuber Growth**  
Nature Communications  
A BibTeX entry will be added once the publisher record is available.



## Contact

Please open an issue in this repository for questions about running the analysis. For requests that cannot be handled publicly, please contact the corresponding author listed in the manuscript.
