"""Truth-only phase-specific LV wall/cavity masks for the static XCAT studies.

Activity distinguishes myocardium (75) from liver (63) and blood pool (2).
The LV is the large enclosed cavity near the canonical phase-zero truth seed;
watershed competitors separate RV/atria. Neither reconstructed images nor DIP
outputs are accepted as inputs. Output axes are gate,x,y,z on the fine ROI.
"""
from __future__ import annotations
import hashlib,io,json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reconstructions'))
from prepare_data import resample_density
from scipy.spatial import ConvexHull,Delaunay
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

SHAPE=np.array([32,32,32])
ORIGIN=np.array([-59.04,-118.08,-38.60307692307692])
STEP=np.array([4.92,4.92,4.825384615384615])
MASK_VERSION='lv_uniform_core_v2'
PARAMETERS=dict(myocardium_native_activity_above=70.,cavity_native_activity_below=15.,
    lv_seed_search_radius_mm=25.,competing_marker_min_radius_mm=5.,
    competing_marker_min_separation_native_voxels=4,
    wall_max_distance_from_lv_cavity_mm=14.,wall_max_distance_from_seed_mm=35.,
    cavity_core_erosion_mm=2.5,cavity_inscribed_radius_fraction=.75,
    seed_centroid_peak_fraction=.9,seed_centroid_neighborhood_mm=8.,
    wall_fine_occupancy_min=.5,cavity_fine_occupancy_min=.75,
    wall_fine_activity_min=65.,cavity_fine_activity_max=8.)


def read_truth(study,gate):
    manifest=json.loads((study/'study.json').read_text())
    path=study/manifest['truth_files'][gate]
    content=path.read_bytes()
    loaded=np.load(io.BytesIO(content),allow_pickle=False)
    native=np.asarray(loaded['data'] if isinstance(loaded,np.lib.npyio.NpzFile) else loaded,dtype=np.float32)
    if isinstance(loaded,np.lib.npyio.NpzFile):loaded.close()
    spacing=np.asarray(manifest['voxel_size_mm'],float)
    full_origin=-np.asarray(native.shape)*spacing/2
    first=np.floor((ORIGIN-full_origin)/spacing+1e-7).astype(int)
    stop=np.ceil((ORIGIN+SHAPE*STEP-full_origin)/spacing-1e-7).astype(int)
    crop=native[tuple(slice(a,b) for a,b in zip(first,stop))].copy()
    origin=full_origin+first*spacing
    coefficient=float(manifest.get('source_activity_reference_mbq',210.))/float(manifest.get('source_reference_native_weight_sum',11681335.))
    density=crop/np.float32(np.prod(spacing/10))
    activity=crop/np.float32(coefficient)
    if not np.isfinite(activity).all() or np.any(activity<0):raise ValueError(f'Invalid source truth: {path}')
    return activity,density,origin,spacing,dict(path=str(path),sha256=hashlib.sha256(content).hexdigest())


def chamber_masks(activity,origin,spacing,anchor=None):
    myo=activity>PARAMETERS['myocardium_native_activity_above']
    points=np.argwhere(myo)
    if len(points)<100:raise ValueError('Insufficient myocardial core in truth ROI')
    hull=ConvexHull(points)
    grid=np.indices(activity.shape).reshape(3,-1).T
    inside=(Delaunay(points[hull.vertices]).find_simplex(grid)>=0).reshape(activity.shape)
    low=inside&(activity<PARAMETERS['cavity_native_activity_below'])
    distance=distance_transform_edt(low,sampling=spacing)
    if anchor is None:
        seed=np.array(np.unravel_index(np.argmax(distance),activity.shape))
    else:
        coords=np.indices(activity.shape).astype(float)
        squared=np.zeros(activity.shape,float)
        for axis in range(3):squared+=(origin[axis]+(coords[axis]+.5)*spacing[axis]-anchor[axis])**2
        selected=(squared<=PARAMETERS['lv_seed_search_radius_mm']**2)&low
        seed=np.array(np.unravel_index(np.argmax(np.where(selected,distance,-1)),activity.shape))
    if distance[tuple(seed)]<5.:raise ValueError('LV cavity seed lacks a 5-mm inscribed sphere')
    peak_radius=float(distance[tuple(seed)])
    coordinates=np.indices(activity.shape).astype(float)
    near_peak=np.zeros(activity.shape,float)
    for axis in range(3):near_peak+=((coordinates[axis]-seed[axis])*spacing[axis])**2
    plateau=(distance>=PARAMETERS['seed_centroid_peak_fraction']*peak_radius)&(near_peak<=PARAMETERS['seed_centroid_neighborhood_mm']**2)
    plateau_points=np.argwhere(plateau)
    seed_mm=np.average(origin+(plateau_points+.5)*spacing,axis=0,weights=distance[plateau]**2)
    radial_squared=np.zeros(activity.shape,float)
    for axis in range(3):radial_squared+=(origin[axis]+(coordinates[axis]+.5)*spacing[axis]-seed_mm[axis])**2
    markers=np.zeros(activity.shape,np.int32);markers[tuple(seed)]=1
    peaks=peak_local_max(distance,min_distance=PARAMETERS['competing_marker_min_separation_native_voxels'],
        threshold_abs=PARAMETERS['competing_marker_min_radius_mm'],exclude_border=False)
    marker_count=1
    for point in peaks:
        if np.linalg.norm((point-seed)*spacing)>PARAMETERS['lv_seed_search_radius_mm']:
            marker_count+=1;markers[tuple(point)]=marker_count
    chambers=watershed(-distance,markers,mask=low)
    cavity=chambers==1
    # Cardiac tissue is assigned to its nearest low-activity chamber. This
    # excludes the RV free wall and atrial walls despite their identical label.
    _,nearest=distance_transform_edt(~low,return_indices=True)
    nearest_chamber=chambers[tuple(nearest)]
    wall=myo&(nearest_chamber==1)&(distance_transform_edt(~cavity,sampling=spacing)<=PARAMETERS['wall_max_distance_from_lv_cavity_mm'])
    wall&=radial_squared<=PARAMETERS['wall_max_distance_from_seed_mm']**2
    core=cavity&(distance_transform_edt(cavity,sampling=spacing)>PARAMETERS['cavity_core_erosion_mm'])
    core&=radial_squared<=(PARAMETERS['cavity_inscribed_radius_fraction']*peak_radius)**2
    if not wall.any() or not core.any():raise ValueError('Empty native LV region')
    return wall,core,seed_mm,dict(native_lv_watershed_region_ml=float(cavity.sum()*np.prod(spacing/10)),
        native_cavity_core_ml=float(core.sum()*np.prod(spacing/10)),cavity_inscribed_sphere_radius_mm=float(PARAMETERS['cavity_inscribed_radius_fraction']*peak_radius),
        native_wall_core_ml=float(wall.sum()*np.prod(spacing/10)),
        lv_seed_inscribed_radius_mm=float(distance[tuple(seed)]),competing_chamber_markers=marker_count-1)


def make_masks(study,gate,anchor):
    activity,density,origin,spacing,provenance=read_truth(study,gate)
    wall_native,cavity_native,seed,qc=chamber_masks(activity,origin,spacing,anchor)
    box=lambda image:resample_density(np.asarray(image,np.float32),origin,spacing,SHAPE,ORIGIN,STEP)
    wall_occupancy=box(wall_native);cavity_occupancy=box(cavity_native)
    fine_activity=box(activity)
    truth=box(density)
    wall=(wall_occupancy>=PARAMETERS['wall_fine_occupancy_min'])&(fine_activity>=PARAMETERS['wall_fine_activity_min'])
    cavity=(cavity_occupancy>=PARAMETERS['cavity_fine_occupancy_min'])&(fine_activity<=PARAMETERS['cavity_fine_activity_max'])
    wall&=~cavity
    if wall.sum()<10 or cavity.sum()<10 or np.any(wall&cavity):
        raise ValueError(f'Invalid fine LV masks gate {gate}: wall={wall.sum()}, cavity={cavity.sum()}')
    qc.update(gate_zero_based=gate,wall_voxels=int(wall.sum()),cavity_voxels=int(cavity.sum()),
        lv_seed_xyz_mm=seed.tolist(),wall_truth_mean_mbq_ml=float(truth[wall].mean()),
        cavity_truth_mean_mbq_ml=float(truth[cavity].mean()),
        wall_truth_activity_min=float(fine_activity[wall].min()),
        cavity_truth_activity_max=float(fine_activity[cavity].max()),fine_wall_core_ml=float(wall.sum()*np.prod(STEP/10)),
        fine_cavity_core_ml=float(cavity.sum()*np.prod(STEP/10)),disjoint=True)
    return wall,cavity,truth,seed,qc,provenance

