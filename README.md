# Quantum Tomography And Belle Test

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-cv2-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-orange)

## Experiment and Course Context
This project is part of **Physics Laboratory C** (undergraduate course) in the School of Physics and Astronomy at **Tel Aviv University**.  
The experiment follows a classical-optical implementation that reproduces the *measurement workflow* used in quantum optics: **density-matrix reconstruction (tomography)** and a **Bell-inequality (CHSH) evaluation**.

**Part A – Tomography:** the density matrix is reconstructed from **three datasets** consisting of **10 / 25 / 50 red-laser pulses**, by extracting coincidence statistics in polarization channels and building the corresponding density-matrix elements.  
**Part B – Bell test:** the **Bell parameter S** is extracted from **16 measurement settings** (a 4×4 grid of analyzer angles). Each setting contains **20 pulses**, measured for different \(\alpha\) and \(\beta\) combinations (\(\alpha,\alpha'\) and \(\beta,\beta'\)) according to the angle table in the reference article and the standard angles used in the lab.

## What the Code Does (Workflow)
The analysis is performed offline on recorded webcam videos of the detector outputs.

1. **ROI definition (once):** the user selects rectangular Regions of Interest (ROIs) on the first frame (Part A: 4 ROIs; Part B: 2 ROIs). ROIs are cached to JSON files to avoid repeated manual selection.
2. **Signal extraction:** for each frame, the code isolates **red intensity** (HSV thresholding for red) inside each ROI and builds a time-series signal per channel.
3. **Pulse detection:** signals are normalized and pulses are detected using `scipy.signal.find_peaks` with configurable threshold and minimum peak distance.
4. **Coincidence counting:** peaks from Alice and Bob channels are matched within a configurable time window to compute coincidence counts.
5. **Part A products:** coincidence counts in the HV basis are converted to two reconstructed density matrices (two ordering conventions are produced).
6. **Part B products:** coincidence counts are computed for all 16 angle settings, plotted as a 4×4 diagnostic grid, and used to calculate the Bell parameter \(S\).

The script can run Part A only, Part B only, or both, and can optionally force ROI re-selection.

## Key Functions (One-Line Descriptions)

### Utility / IO
- **_now()**  
  Returns a timestamp string used for logs and report headers.

- **save_roi_cache(path, video_path, labels, rois, cfg)**  
  Writes selected ROIs to a JSON cache file (including labels and resize factor) for reuse across runs.

- **load_roi_cache(path, expected_labels)**  
  Loads ROIs from a JSON cache if it exists and matches the expected ROI labels; otherwise returns `None`.

- **append_report(path, lines)**  
  Appends formatted diagnostic lines to a text quality report file.

### Signal Processing
- **normalize_signal(x)**  
  Converts a raw signal to a \([0,1]\) normalized signal using min–max scaling (returns zeros if the signal is flat).

- **red_sum_in_roi(frame_bgr, roi)**  
  Computes the total red-mask intensity inside an ROI using HSV thresholding for red pixels.

- **first_red_index(sig, thr, max_frames)**  
  Finds the first frame index where the signal crosses a red-intensity threshold (used for alignment/start trimming).

- **coincidences(peaks_a, peaks_b, window)**  
  Counts one-to-one matched peak pairs between two peak lists within a given frame-distance window.

### ROI Selection (GUI)
- **select_roi_white(win, img_bgr)**  
  Provides an interactive OpenCV GUI to draw a rectangular ROI (white rectangle) and returns \((x,y,w,h)\).

- **select_rois_once(video_path, labels, cfg)**  
  Opens the first frame of a reference video and collects ROIs for all requested labels (scaled back to original resolution).

- **get_rois_with_cache(cache_path, video_path, labels, cfg, force_reselect)**  
  Loads ROIs from cache when available; otherwise triggers GUI selection once and saves the result to cache.

## Part A (Tomography)
- **read_signals_nroi(video_path, rois, cfg)**  
  Extracts red-intensity time-series for all four ROIs, aligns them by a detected start index, and trims to equal length.

- **peaks_from(sig, cfg)**  
  Normalizes a signal and extracts pulse indices using `find_peaks` with the configured threshold and minimum distance.

- **counts_HV_basis(peaks4, cfg, bob_swap=False)**  
  Converts four peak lists (A_V, A_H, B_V, B_H) into coincidence counts \((N_{HH}, N_{HV}, N_{VH}, N_{VV})\), with optional Bob channel swap.

- **rho_phi(N_HH, N_HV, N_VH, N_VV)**  
  Builds a 4×4 density matrix emphasizing the \(|HH\rangle\) and \(|VV\rangle\) populations with a coherence term estimated from probabilities.

- **rho_psi(N_HH, N_HV, N_VH, N_VV)**  
  Builds a 4×4 density matrix emphasizing the \(|HV\rangle\) and \(|VH\rangle\) populations with a coherence term estimated from probabilities.

- **plot_rho_3d(ax, rho, title, cfg)**  
  Produces a 3D bar visualization of \(|\rho_{ij}|\) with a colormap and saves a colorbar for interpretation.

- **run_part_a(cfg, force_reselect_roi)**  
  Runs Part A end-to-end: loads videos, selects/loads ROIs, extracts signals, detects peaks, computes coincidence counts, reconstructs two density matrices per dataset, and saves the summary figure.

## Part B (Bell Parameter)
- **read_signals_2roi(video_path, rois2, cfg)**  
  Extracts and aligns red-intensity signals for the two ROIs (Alice, Bob) from a given measurement video.

- **canon_alpha(a, cfg)**  
  Maps a given \(\alpha\) angle into the nearest canonical \(\alpha\) value used in the experiment (handles wrap-around).

- **canon_beta(b, cfg)**  
  Maps a given \(\beta\) angle into the nearest canonical \(\beta\) value used in the experiment (handles wrap-around conventions).

- **run_part_b(cfg, force_reselect_roi)**  
  Runs Part B end-to-end: loads ROIs, processes all 16 videos, counts coincidences per \((\alpha,\beta)\), saves a 4×4 diagnostic plot grid, computes correlators \(E(\alpha,\beta)\), and prints the Bell parameter \(S\).

## Command-Line Interface
- **parse_args()**  
  Parses CLI flags to control execution (Part A only / Part B only, disable plotting, force ROI re-selection).

Run examples:
```bash
python your_script.py
python your_script.py --only-a
python your_script.py --only-b
python your_script.py --no-show
python your_script.py --reselect-roi-a
python your_script.py --reselect-roi-b 
