# Quantum Tomography And Bell Test

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-cv2-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-orange)

## Experiment and Course Context
This project is part of **Physics Laboratory C** (undergraduate course) in the School of Physics and Astronomy at **Tel Aviv University**.  
The experiment follows a classical-optical implementation that reproduces the *measurement workflow* used in quantum optics: **density-matrix reconstruction (tomography)** and a **Bell-inequality (CHSH) evaluation**.

**Part A – Tomography:** the density matrix is reconstructed from **three datasets** consisting of **10 / 25 / 50 red-laser pulses**, by extracting coincidence statistics in polarization channels and building the corresponding density-matrix elements.  

**Part B – Bell test:** the **Bell parameter S** is extracted from **16 measurement settings** (a 4×4 grid of analyzer angles). Each setting contains **20 pulses**, measured for different analyzer-angle combinations \((\alpha,\alpha')\) and \((\beta,\beta')\), according to the angle table in the reference article and the standard angles used in the laboratory experiment.

---

## What the Code Does (Workflow)
The analysis is performed offline on recorded webcam videos of the detector outputs.

1. **ROI definition (once):** the user selects rectangular Regions of Interest (ROIs) on the first frame  
   (Part A: 4 ROIs; Part B: 2 ROIs). ROIs are cached to JSON files to avoid repeated manual selection.
2. **Signal extraction:** for each frame, the code isolates **red intensity** (HSV thresholding for red) inside each ROI and builds a time-series signal per channel.
3. **Pulse detection:** signals are normalized and pulses are detected using `scipy.signal.find_peaks` with configurable threshold and minimum peak distance.
4. **Coincidence counting:** peaks from Alice and Bob channels are matched within a configurable time window to compute coincidence counts.
5. **Part A products:** coincidence counts in the HV basis are converted to reconstructed density matrices.
6. **Part B products:** coincidence counts are computed for all 16 angle settings, plotted as a 4×4 diagnostic grid, and used to calculate the Bell parameter \(S\).

The script can run Part A only, Part B only, or both, and can optionally force ROI re-selection.

---

## Inputs
The analysis relies on the following inputs:

1. **Experimental video files**
   - **Part A:** video files named `partbXXbits.mp4` (e.g. 10, 25, 50 bits) located in the directory specified by `CFG.data_dir_a`.
   - **Part B:** video files named `bit1.mp4` to `bit16.mp4` located in `CFG.data_dir_b`.

2. **Manual ROI selection**
   - During the first run, the user selects regions of interest (ROI) using an interactive OpenCV GUI.
   - The selected ROIs are saved to cache files (`roi_cache_partA.json`, `roi_cache_partB.json`) and reused in subsequent runs unless re-selection is forced.

3. **Algorithm parameters**
   - Signal thresholds, peak-detection parameters, coincidence window size, and analyzer angles are defined in the `Config` dataclass and control the full analysis flow.

4. **Command-line flags**   ****Check it***
   - Optional CLI arguments allow running only Part A or Part B, disabling plot display, or forcing ROI re-selection.

---

## Key Functions

### Utility / IO
- **_now()** – returns a timestamp string for logs and reports.  
- **save_roi_cache(...)** – saves selected ROIs to a JSON cache file.  
- **load_roi_cache(...)** – loads cached ROIs if they match the expected labels.  
- **append_report(...)** – appends diagnostic information to a text report file.

### Signal Processing
- **normalize_signal(x)** – rescales a signal to the polirazer ( 0 or 1 ).  
- **red_sum_in_roi(frame, roi)** – computes red-intensity sum inside an ROI using HSV masking.  
- **first_red_index(sig, thr, max_frames)** – finds the first frame containing a significant red signal.  
- **coincidences(peaks_a, peaks_b, window)** – counts matched coincidence events within a frame window.

### ROI Selection
- **select_roi_white(...)** – interactive GUI for drawing a rectangular ROI.  
- **select_rois_once(...)** – collects ROIs from the first frame of a reference video.  
- **get_rois_with_cache(...)** – loads ROIs from cache or triggers manual selection.

---

## Part A – Quantum Tomography And Dentisy Matrix
- **read_signals_nroi(...)** – extracts and aligns red-intensity signals from four ROIs.  
- **peaks_from(...)** – detects pulse peaks in a normalized signal.  
- **counts_HV_basis(...)** – computes coincidence counts in the HV polarization basis.  
- **rho_phi(...)** – reconstructs a density matrix dominated by \(|HH\rangle\) and \(|VV\rangle\).  
- **rho_psi(...)** – reconstructs a density matrix dominated by \(|HV\rangle\) and \(|VH\rangle\).  
- **plot_rho_3d(...)** – visualizes the absolute density-matrix elements in 3D.  
- **run_part_a(...)** – executes the full tomography pipeline and saves density-matrix plots.

---

## Part B – Bell Inequality Test
- **read_signals_2roi(...)** – extracts aligned signals for Alice and Bob detectors.  
- **canon_alpha(...)** – maps analyzer angle \(\alpha\) to the nearest canonical value.  
- **canon_beta(...)** – maps analyzer angle \(\beta\) to the nearest canonical value.  
- **run_part_b(...)** – processes all 16 measurements, computes correlations, and evaluates the Bell parameter \(S\).

---

## Outputs
The code produces the following outputs:

- **Figures**
  - `PartA_density_matrices.png` – 3D visualizations of reconstructed density matrices.
  - `PartB_grid.png` – 4×4 grid of normalized signals for all Bell-test angle settings.

- **Text reports**
  - `quality_report_partA.txt` – diagnostic information for tomography runs.
  - `quality_report_partB.txt` – diagnostic information for Bell-test runs.

- **Console output**
  - Printed density matrices (Part A).
  - Printed Bell parameter \(S\) (Part B).

---

## Command-Line Interface
- **parse_args()** – parses CLI flags to control execution.

Run examples:
```bash
python your_script.py
python your_script.py --only-a
python your_script.py --only-b
python your_script.py --no-show
python your_script.py --reselect-roi-a
python your_script.py --reselect-roi-b

