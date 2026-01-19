# SWAMP

SWAMP is a research prototype for non invasive subsurface sensing of sweet potato growth and localization using wireless signals. The system combines two measurement modalities.

1. Channel Frequency Response (CFR) sweeps collected with SDRs.
2. LTE based link and channel quality metrics collected with srsRAN.

This repository provides code that can be shared at this time for data processing, feature extraction, and analysis used in the paper draft.

## Repository status

The public release is currently partial.

1. Analysis code is included.
2. Some experiment control code is not included because it is tightly coupled to lab specific SDR hardware setup and local deployment configuration.
3. If you need the missing components for verification, please contact the authors with a brief description of your intended use.

## Quick start

1. Clone this repository.
2. Create a Python environment.
3. Install Python dependencies.
4. Run a small smoke test on the example scripts.

Example commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "print('swamp env ok')"
```

## Dependencies

### Python

Use Python 3.10 or newer.

Common packages used in analysis include numpy, scipy, pandas, matplotlib, scikit learn.

### SDR and RF stack

If you are running the measurement stack rather than only analysis, you will also need the SDR driver and RF software stack used by the paper.

1. UHD drivers for USRP X310.
2. GNU Radio.
3. srsRAN Release v22.04, modified in our internal branch.

If you only want to reproduce plots from exported measurements, you do not need UHD, GNU Radio, or srsRAN.

## Suggested directory layout

If your repository already uses a different structure, keep it. If not, this is a simple structure that works well.

1. `analysis/`  
   Plotting, statistics, model fitting, evaluation scripts.

2. `features/`  
   CFR metric extraction and LTE metric extraction utilities.

3. `data/`  
   Not tracked by git. Place raw or processed datasets here.

4. `configs/`  
   Example configuration files for experiments or processing.

## Reproducing key results

High level pipeline used in the paper.

1. Load CFR sweep data.
2. Compute CFR metrics such as BAI, H over L, Slope, PhaseRMS, RippleVar, EchoTail.
3. Load LTE derived metrics such as RSRP, SINR, MCS, BLER, throughput.
4. Train or fit fusion weights on the training subset.
5. Evaluate on the held out subset and generate plots.

Scripts and exact entry points depend on the current repository snapshot. Start from the top level analysis script in `analysis/` and follow the imports.

## Data availability

Measurement datasets are not included in the public repository at the moment.

1. Some datasets contain lab specific metadata and are being cleaned for release.
2. If you are a reviewer or editor requesting access, contact the authors and we can provide an archived bundle suitable for verification.

## Code availability

This repository contains the code used in the study where it can be shared at this time, including analysis and feature extraction. Some experiment control components depend on lab specific SDR hardware configuration and are not included in the public release. Additional code needed to reproduce the experiments end to end can be made available from the authors upon reasonable request.

## Citation

If you use this code in academic work, please cite the SWAMP paper once it is public. A bibtex entry will be added when the preprint or publication link is available.

## Contact

Open an issue in this repository for questions about running the analysis. For access requests that cannot be handled publicly, contact the corresponding author listed in the manuscript.
