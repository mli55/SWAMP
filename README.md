# SWAMP

Code release for the paper **Commodity Wireless Links Monitor Belowground Tuber Growth**  

## Overview

SWAMP is a research prototype for noninvasive subsurface sensing of belowground tuber growth and localization using wireless signals. The study combines two measurement modalities.

- Channel Frequency Response (CFR) sweeps collected with software defined radios  
- LTE based link and channel quality indicators collected with an srsRAN based private LTE setup

This repository contains the portions of the codebase that can be shared at this time for data processing, feature extraction, and analysis.

## Repository structure

Current top level directories in this snapshot

- `cfr_gnuradio/`  
  CFR related scripts, processing utilities, and supporting assets.

- `cfr_gnuradio/`  
  srsRAN based LTE stack with project specific modifications used for metric logging.

- `simulation/`  
  Simulation workspace with a Python package, scripts, and tests.

## What this release includes

- Analysis and figure generation scripts.
- Feature extraction utilities for CFR derived metrics and LTE derived indicators.
- Simulation utilities where applicable.

Some experiment control and deployment components that are tightly coupled to lab specific SDR hardware setup and local network configuration are not included.

## Citation

If you use this work, please cite the paper:

**Commodity Wireless Links Monitor Belowground Tuber Growth**  
Nature Communications  
A BibTeX entry can be added once the publisher record is available.

## Contact

Please open an issue in this repository for questions about running the analysis. For requests that cannot be handled publicly, please contact the corresponding author listed in the manuscript.
