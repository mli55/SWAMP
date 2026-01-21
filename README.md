# SWAMP

SWAMP is a research prototype for noninvasive subsurface sensing of sweet potato growth and localization using wireless signals. The study uses two measurement modalities.

1. Channel Frequency Response (CFR) sweeps collected with software defined radios.
2. LTE based link and channel quality metrics collected with srsRAN.

This repository provides code that can be shared at this time for data processing, feature extraction, simulation, and analysis used in the manuscript.

## Repository layout

Top level components

- `GNU/`  
  Analysis scripts and utilities used to process CFR and LTE metric logs. This folder also contains example measurement exports and example figures.

- `simu/`  
  A self contained simulation workspace, including a small Python package under `simu/rfspsim/`, scripts, and tests.

- `PSI_SRS/`  
  A vendor tree based on srsRAN with project specific modifications and supporting files.

## Repository status

The public release is currently partial.

Included

- Analysis and feature extraction scripts.
- Simulation code and tests.
- The modified RF and LTE stack sources that were used for metric logging, when they can be shared.

Not included

- Some experiment control and deployment components that are tightly coupled to lab specific SDR hardware setup and local configuration.
- Full raw datasets used in the paper.

If you need missing components for verification, contact the authors with a brief description of the intended use.

## Requirements

### Python

A recent Python version is recommended. The `simu/` folder includes its own dependency specification.

### RF stack

If you are running the measurement stack rather than only analysis, you will need an SDR driver and RF software stack compatible with the hardware used in the study. If you only want to reproduce plots from exported measurements, you do not need an SDR driver or a full LTE stack build.

## Reproducing key results

High level pipeline used in the manuscript

1. Load CFR sweep data.
2. Compute CFR derived metrics used in the study.
3. Load LTE derived metrics from srsRAN logs.
4. Fit or train the fusion model on the training subset.
5. Evaluate on the held out subset and generate plots.

Entry points vary across snapshots. A practical starting point is the analysis scripts under `GNU/`, then follow imports and comments to the supporting utilities and the example inputs.

## Data availability

Measurement datasets are not included in the public repository at the moment.

Some datasets contain lab specific metadata and are being cleaned for release. If you are a reviewer or editor requesting access, contact the authors and we can provide an archived bundle suitable for verification.

## Code availability

Code used in this study is available in this repository where it can be shared at this time, including analysis, feature extraction, simulation, and the modified srsRAN based components used to collect LTE link and channel quality metrics. Some experiment control components depend on lab specific SDR hardware configuration and local deployment details and are not included in the public release. Additional code and artifacts needed to reproduce the end to end experiments can be made available from the authors upon reasonable request.

## Licensing

This repository includes components under different licenses. See license files within each subdirectory, including `PSI_SRS/LICENSE`, and follow any upstream attribution requirements.

## Citation

If you use this code in academic work, please cite the SWAMP paper once it is public. A BibTeX entry will be added when a preprint or publication link is available.

## Contact

Open an issue in this repository for questions about running the analysis. For access requests that cannot be handled publicly, contact the corresponding author listed in the manuscript.