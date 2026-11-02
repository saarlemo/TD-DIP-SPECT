# Time-dependent deep image prior for cardiac SPECT reconstruction

## Getting started
### Software requirements
- XCAT
  - The ```XCAT_PATH``` environment variable should point to the directory containing dxcat2_64bit and the XCAT configuration and phantom files, including xcat2.cfg and vfemale50.nrb
- SIMIND v8.0
  - The ```SIMIND_PATH``` environment variable should point to the directory containing the simind executable and the smc_dir data directory.
- [OMEGA toolbox](https://github.com/saarlemo/OMEGA/tree/spect)
- Tested on Linux (Ubuntu 24.04)

### Running the suite


## TD-DIP

In the table below is a brief review of DIP in medical image reconstruction. Publications on denoising are omitted from the list on purpose.

| Method                 | Modality |       Network architecture        | Temporal features | Dimensionality |
| ---------------------- | :------: | :-------------------------------: | :---------------: | :------------: |
|   Our implementation   |  SPECT   | FCNN + decoder (multi-resolution) | Latent coordinate |       3D       |
|    Yoo *et al.* [1]    |   MRI    |          FCNN + decoder           | Latent coordinate |       2D       |
|    Wu *et al.* [2]     |  SPECT   |  Multiple U-net, multiple FCNN    |       INR         |       2D       |
|    Hori *et al.* [3]   |  SPECT   |              U-net                |       None        |       2D       |
| Hashimoto *et al.* [4] |   PET    |              U-net                |       None        |       3D       |


## ECG gating schemes
### Conventional "static" gating
### Cycle-resolved "dynamic" gating


## Dataset
### XCAT phantom
The XCAT numerical phantom was used to generate activity and attenuation volumes for one cardiac cycle of one second, sampled at 256 equally spaced cardiac phases. The phantom includes a beating heart with a fixed tracer distribution; respiratory motion and changes in tracer uptake are not simulated. Each volume has a matrix size of 256 × 256 × 306 voxels and an isotropic voxel size of 2.46 mm. The 752.76 mm axial coverage extends approximately 25% beyond each end of the 500 mm detector head.

### SIMIND simulations
The simulated acquisition represents a 20-minute ECG-gated myocardial perfusion SPECT study using a dual-head camera with a 90° angular offset between its detector heads. The activity distribution represents a Tc99m-like tracer, with 140.5 keV photons and an emission abundance of 0.891. The two heads acquire data simultaneously while moving through 120 positions over a full 360° rotation, with 10 seconds of acquisition at each position.

SIMIND is run separately for each of the 256 cardiac phases, using the corresponding activity and attenuation volumes. Each phase is simulated for both detector heads, resulting in 512 independent simulations. For each phase and head, projections are generated at all 120 detector positions, representing the measurements accumulated at that phase during the acquisition. At each position, the exposure is 10/256 seconds per phase and head; summed over all 256 phases, this gives 10 seconds per position for each head. As one cardiac cycle is one second, a singular phase within any one cardiac cycle is acquired by Poisson thinning with equal probability of 1/10 for each cycle.

The camera model uses a physical crystal measuring 500 mm axially and 400 mm transversely and a GE Infinia GI-LEHR collimator. The 128 × 128 projection grid has 3.5 mm pixels and therefore covers 448 × 448 mm: it crops 26 mm from each axial end of the crystal and extends 24 mm beyond the crystal on each transverse side. In the study's reconstruction processing, projection pixels outside the physical transverse detector width are masked. The detectors follow a body-contour orbit with a 2 cm air gap. The simulated energy resolution is 9% at 140 keV and the intrinsic spatial resolution is 3.8 mm FWHM. Each phase/head simulation uses approximately 500,000 Monte Carlo photon histories per view and an independent random-number stream. The activity concentration is based on 210 MBq in a 472.32 mm reference thoracic region; because the simulated volume extends farther axially, its total activity is larger, approximately 247.5 MBq at the first phase.

Three energy windows are saved: a 126.45–154.55 keV photopeak window, a 119.425–126.45 keV lower scatter window, and a 154.55–161.575 keV upper scatter window. In the study's subsequent processing, the two side windows are used for triple-energy-window scatter estimation, with the estimate calculated as twice their sum.


## References
[1] Yoo, Jaejun, Kyong Hwan Jin, Harshit Gupta, Jerome Yerly, Matthias Stuber, and Michael Unser. 2021. “Time-Dependent Deep Image Prior for Dynamic MRI.” IEEE Transactions on Medical Imaging 40 (12): 3337–48. https://doi.org/10.1109/tmi.2021.3084288.

[2] Wu, Yicheng, Roy Y. He, Qiaoqiao Ding, Xiaoqun Zhang, and Chao Wang. 2026. ‘Robust Dynamic SPECT Reconstruction with Scarce Angular and Limited Temporal Sampling’. Journal of Mathematical Imaging and Vision 68 (3). https://doi.org/10.1007/s10851-026-01302-z.

[3] Hori, Kensuke, Fumio Hashimoto, Kazuya Koyama, and Takeyuki Hashimoto. 2025. “Limited-Angle SPECT Image Reconstruction Using Deep Image Prior.” Physics in Medicine & Biology 70 (14): 145005. https://doi.org/10.1088/1361-6560/adea09.

[4] Hashimoto, Fumio, Yuya Onishi, Kibo Ote, Hideaki Tashima, and Taiga Yamaya. 2023. “Fully 3D Implementation of the End-to-End Deep Image Prior-Based PET Image Reconstruction Using Block Iterative Algorithm.” Physics in Medicine & Biology 68 (15): 155009. https://doi.org/10.1088/1361-6560/ace49c.
