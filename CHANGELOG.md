# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 8.0.0

### Breaking changes

- `ZebrafishHMM_G4_Sym` now has a `σforw` parameter, instead of a `Normal` distribution

### Added

- `save_hmm` and `load_hmm` for `ZebrafishHMM_G3_Sym` and `ZebrafishHMM_G4_Sym`

## 7.4.0

### Added

- `chunks` to split a trajectory into chunks of similar size.

## 7.2.0

### Added

- `legoc2021_single_fish_T26_trajs()`, loads long single fish trajectories from Le Goc et al 2021.

## 7.0.0

### Breaking changes

- Changed stubborness factor definition, shifting `q` by one, to be consistent with the paper. `stubborness_factor(hmm, q)` now gives the value corresponding to `stubborness_factor(hmm, q - 1)` in previous versions.

## 4.0.0

### Breaking changes

- Renamed models to `ZebrafishHMM_G3` and `ZebrafishHMM_G4` (for 3 and 4 states models, with Gamma turn angle emissions).
- No longer using `DistributionMissingWrapper`. Trajectories have missing values (`NaN`) only at the end, for padding. We can just remove these missing entries, instead of using the `DistributionMissingWrapper` trick. 

### Added

- `load_behaviour_free_swimming_trajs`, to load a `Vector` of trajectories (which are also `Vector`s), after filtering out missing entries.
- Truncated normal models.