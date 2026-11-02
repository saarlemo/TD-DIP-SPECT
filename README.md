# Time-dependent deep image prior for cardiac SPECT reconstruction

## Getting started
### Software requirements
- XCAT (env variable path)
- SIMIND v8.0 (env variable path)
- OMEGA toolbox
- Tested on Linux (Ubuntu 24.04)

### Running the suite


## Detailed description
### XCAT phantom
The XCAT numerical phantom was used to generate activity and attenuation volumes for one cardiac cycle of one second, sampled at 256 equally spaced cardiac phases. The phantom includes a beating heart with a fixed tracer distribution; respiratory motion and changes in tracer uptake are not simulated. Each volume has a matrix size of 256 × 256 × 306 voxels and an isotropic voxel size of 2.46 mm. The 752.76 mm axial coverage extends approximately 25% beyond each end of the 500 mm detector head.

### SIMIND simulations
The simulated acquisition represents a 20-minute ECG-gated myocardial perfusion SPECT study using a dual-head camera with a 90° angular offset between its detector heads. The activity distribution represents a Tc99m-like tracer, with 140.5 keV photons and an emission abundance of 0.891. The two heads acquire data simultaneously while moving through 120 positions over a full 360° rotation, with 10 seconds of acquisition at each position.

SIMIND is run separately for each of the 256 cardiac phases, using the corresponding activity and attenuation volumes. Each phase is simulated for both detector heads, resulting in 512 independent simulations. For each phase and head, projections are generated at all 120 detector positions, representing the measurements accumulated at that phase during the acquisition. At each position, the exposure is 10/256 seconds per phase and head; summed over all 256 phases, this gives 10 seconds per position for each head. As one cardiac cycle is one second, a singular phase within any one cardiac cycle is acquired by Poisson thinning with equal probability of 1/10 for each cycle.

The camera model uses a physical crystal measuring 500 mm axially and 400 mm transversely and a GE Infinia GI-LEHR collimator. The 128 × 128 projection grid has 3.5 mm pixels and therefore covers 448 × 448 mm: it crops 26 mm from each axial end of the crystal and extends 24 mm beyond the crystal on each transverse side. In the study's reconstruction processing, projection pixels outside the physical transverse detector width are masked. The detectors follow a body-contour orbit with a 2 cm air gap. The simulated energy resolution is 9% at 140 keV and the intrinsic spatial resolution is 3.8 mm FWHM. Each phase/head simulation uses approximately 500,000 Monte Carlo photon histories per view and an independent random-number stream. The activity concentration is based on 210 MBq in a 472.32 mm reference thoracic region; because the simulated volume extends farther axially, its total activity is larger, approximately 247.5 MBq at the first phase.

Three energy windows are saved: a 126.45–154.55 keV photopeak window, a 119.425–126.45 keV lower scatter window, and a 154.55–161.575 keV upper scatter window. In the study's subsequent processing, the two side windows are used for triple-energy-window scatter estimation, with the estimate calculated as twice their sum.

### ECG gating schemes
#### Conventional "static" gating
#### Cycle-resolved "dynamic" gating

### Reconstructions
#### OSEM
#### BSREM + NLM
#### TD-DIP

### Plotting the metrics


## References


