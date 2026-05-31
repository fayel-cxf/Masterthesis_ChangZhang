# Marine Bioinvasion Risk Framework — New Zealand

Code repository for the master thesis *From Synthetic Trajectory Generation to Voyage-Level Risk:A Biosecurity Assessment Framework for New Zealand Maritime Arrivals*.

## Overview

The framework estimates the probability of non-indigenous species introduction to New Zealand ports through ballast water discharge. It combines two track synthetic voyage generation: smoothed markov chain based generator (Track 1) & a CVAE-based synthetic voyage generator (Track 2) to extend the risk corpus beyond the observed dataset.

Risk is computed as a three-layer product:

**P_invasion = P_alien × P_intro × P_estab**

| Layer | Description |
|-------|-------------|
| P_alien | Biogeographic distance between source and NZ province |
| P_intro | Ballast water volume and organism survival en route |
| P_estab | Environmental similarity between source and NZ ecoprovince |

## Repository Structure

```
├── data/                          # Input data (restricted files excluded)
│   ├── ecoprov_envdist_scaled.csv # Environmental similarity matrix (Seebens et al.)
│   └── track2_synthetic_voyages.csv
├── output/                        # All script outputs
├── figure/                        # Generated figures
├── data_preparation.py            # Raw data filtering and sequential format conversion
│
├── cluster/                       # Voyage clustering (k-means on route features)
│   ├── cluster.m1.py
│   ├── cluster.m1.kmeans.py
│   └── voyage_cluster_map.py
│
├── dwt/                           # DWT scraping from Equasis
│   └── dwt.py
│
├── track1_markovchain/            # Track 1 — Markov chain synthetic voyage generation
│   ├── track1_markovchain_code/
│   │   ├── eda/eda_observed_voyages.py
│   │   ├── mc_generator/markov_voyage_generator.py  # Markov chain generator
│   │   ├── seebens3/risk_calculation_mc.py           # Three-layer risk (MC voyages)
│   │   └── validation/                               # Layer 1/2/3 validation scripts
│   └── data/
│       ├── markov/mc.syn.all.csv
│       ├── markov/mc.syn.all.layer2.csv
│       └── seebens/track1_risk_results.csv
│
├── quantitative risk framework/   # Shared risk computation and post-assessment
│   ├── risk_calculation.py        # Three-layer risk for observed + Track 2 synthetic voyages
│   ├── track2_layer3_post_assessment.py  # Validation statistics and gap analysis
│   ├── make_figures.py            # Publication figures
│   ├── pooling_port_ranking_analysis.py  # Port ranking under pooled corpus
│   └── dwt_log_fit/               # Ballast water regression (DWT → W_r)
│
└── track_2_CVAE/                  # Track 2 — CVAE synthetic voyage generation
    ├── config.py                  # Shared paths and hyperparameters
    ├── run_phase1.py              # CVAE training + Phase 1 generation
    ├── run_phase2.py              # Phase 2 generation with Layer 2 filter
    ├── model/cvae.py              # Conditional VAE architecture
    ├── data/{dataset,preprocess}.py
    ├── generation/{sample,filter}.py
    ├── training/train.py
    └── evaluation/                # Speed threshold derivation and validation
```

## Execution Order

```bash
python data_preparation.py                                                        # Step 1: prepare data
python cluster/cluster.m1.py                                                      # Step 2: cluster voyages
python dwt/dwt.py                                                                 # Step 3: scrape DWT from Equasis
python "quantitative risk framework/dwt_log_fit/fit_log_regression.py"           # Step 4: fit DWT regression
python track1_markovchain/track1_markovchain_code/mc_generator/markov_voyage_generator.py  # Step 5: generate Track 1 synthetic voyages
python track1_markovchain/track1_markovchain_code/seebens3/risk_calculation_mc.py          # Step 6: compute Track 1 risk
cd track_2_CVAE && python run_phase1.py && python run_phase2.py                   # Step 7: generate Track 2 synthetic voyages
cd ..
python "quantitative risk framework/risk_calculation.py"                          # Step 8: compute observed + Track 2 risk
python "quantitative risk framework/track2_layer3_post_assessment.py"             # Step 9: validate
python "quantitative risk framework/pooling_port_ranking_analysis.py"             # Step 10: pooled corpus analysis
python "quantitative risk framework/make_figures.py"                              # Step 11: produce figures
```

## Data

Raw AIS vessel event data is provided by an institutional partner and is not included in this repository. Public-source files (environmental similarity matrix, regression results) are included in `data/`.

## Requirements

```bash
pip install -r requirements.txt
```

## Authors

Xiaofei Chang & Yunlu Zhang — KU Leuven, Master Thesis 2026
