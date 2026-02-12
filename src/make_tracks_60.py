import os
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import json, time, subprocess
import re
from datetime import datetime

INPUTS = {
    # Required in practice
    "OUTPUT_FOLDER": "",
    "NUM_TRAIN_TRACKS": 5,
    "NUM_TEST_TRACKS": 5,

    # Detector geometry
    "NUMBER_OF_LAYERS": 25,
    "DETECTOR_LENGTH": 320.0,
    "SMALLEST_LAYER": 3.1,
    "LARGEST_LAYER": 53.0,

    # Fourier settings
    "FOURIER_DIM_TRAIN": 25,
    "FOURIER_DIM_TEST": 25,
    "TRAIN_FUNCTION": 3,
    "TEST_FUNCTION": 3,

    # Train/test split behavior
    "DISJOINT": False,

    # Hit noise
    "ADD_HIT_NOISE": True,
    "WRITE_HIT_SIGMAS": True,
    "HIT_NOISE_SIGMA_XY": 0.01,
    "HIT_NOISE_SIGMA_Z": 0.01,
    
    # Standard Model?
    "STANDARD_MODEL": False
}

# Physical Constants
_C = 2.99792458e8               # speed of light, m/s
_E_CHARGE = 1.602176634e-19     # elementary charge, C
# Conversion factor: 1 (GeV/c) -> kg*m/s  (use E/c where 1 GeV = 1.602176634e-10 J)
_GEV_C_TO_KGMS = (1.602176634e-10) / _C   # ≈ 5.344286e-19 kg·m/s

def slug(x):
    """Filesystem-safe short token."""
    s = str(x)
    s = s.replace(" ", "")
    s = s.replace(".", "p")
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_\-p]", "", s)
    return s

def make_dataset_name(INPUTS):
    # Pull key knobs (with defaults matching your script)
    ntrain = INPUTS.get("NUM_TRAIN_TRACKS", "NA")
    ntest  = INPUTS.get("NUM_TEST_TRACKS", "NA")

    nlayers = INPUTS.get("NUMBER_OF_LAYERS", 25)
    detL    = INPUTS.get("DETECTOR_LENGTH", 320.0)
    rmin    = INPUTS.get("SMALLEST_LAYER", 3.1)
    rmax    = INPUTS.get("LARGEST_LAYER", 53.0)

    fd_tr = INPUTS.get("FOURIER_DIM_TRAIN", 25)
    fd_te = INPUTS.get("FOURIER_DIM_TEST", 25)

    f_tr  = INPUTS.get("TRAIN_FUNCTION", 3)
    f_te  = INPUTS.get("TEST_FUNCTION", 3)

    add_noise = bool(INPUTS.get("ADD_HIT_NOISE", False))
    sig_xy = INPUTS.get("HIT_NOISE_SIGMA_XY", 0.01)
    sig_z  = INPUTS.get("HIT_NOISE_SIGMA_Z", 0.01)

    # Name
    parts = [
        f"train{slug(ntrain)}_test{slug(ntest)}",
        f"layers{slug(nlayers)}_len{slug(detL)}",
        f"r{slug(rmin)}-{slug(rmax)}",
        f"fd{slug(fd_tr)}-{slug(fd_te)}",
        f"func{slug(f_tr)}-{slug(f_te)}",
    ]
    if add_noise:
        parts.append(f"noiseXY{slug(sig_xy)}_Z{slug(sig_z)}")
    else:
        parts.append("noiseless")
    if INPUTS.get("STANDARD_MODEL", False):
        parts.append("standardModel")

    return "v" + datetime.now().strftime("%Y%m%d_%H%M%S") + "__" + "__".join(parts)

#user defined input
num_layers = INPUTS.get('NUMBER_OF_LAYERS', 25)
detector_length = INPUTS.get('DETECTOR_LENGTH', 320)
smallest_layer = INPUTS.get('SMALLEST_LAYER', 3.1)
largest_layer = INPUTS.get('LARGEST_LAYER', 53)
#If you manually set 'OUTPUT_FOLDER' in INPUTS, it uses that.
#Otherwise it auto-generates.
new_output_folder = INPUTS.get('OUTPUT_FOLDER') or make_dataset_name(INPUTS)
fourierDimTest = INPUTS.get('FOURIER_DIM_TEST', 25)
fourierDimTrain = INPUTS.get('FOURIER_DIM_TRAIN', 25)
train_size = INPUTS.get('NUM_TRAIN_TRACKS')
test_size = INPUTS.get('NUM_TEST_TRACKS')
schwartz_train_function = INPUTS.get('TRAIN_FUNCTION', 3)
schwartz_test_function = INPUTS.get('TEST_FUNCTION', 6)
# --- Hit measurement noise (cm) ---
HIT_NOISE_SIGMA_XY = float(INPUTS.get("HIT_NOISE_SIGMA_XY", 0.01))  # 100 microns
HIT_NOISE_SIGMA_Z  = float(INPUTS.get("HIT_NOISE_SIGMA_Z",  0.01))  # 100 microns
ADD_HIT_NOISE      = INPUTS.get("ADD_HIT_NOISE", False)
WRITE_HIT_SIGMAS   = INPUTS.get("WRITE_HIT_SIGMAS", True)
# Standard Model?
STANDARD_MODEL = INPUTS.get("STANDARD_MODEL", False)

ATLASradii = np.linspace(smallest_layer, largest_layer, num_layers)

validate_size = 0

fourierDimValidate = 1

intersection_of_sets = 4
signal_PID = 15
bField = 1
max_B_radius = np.max(ATLASradii)
Schwartz_train_test_on_disjoint_Sets = INPUTS.get('DISJOINT', False)
start_train_dim = 0
start_test_dim = 4
output_dir = '/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed'

#Don't change this. Bc of how loops work Dim = n would really be interpreted as Dim = n-1 without this
#Bc range(n) is not inclusive of n
fourierDimTrain += 1
fourierDimTest += 1
fourierDimValidate += 1

signal_tracks_per_event = 1

schwartz_1 = [60 * np.exp(-0.01 * (n ** 2)) for n in range(fourierDimTrain)]
schwartz_2 = [30 * np.exp(-0.01 * (n ** 2)) for n in range(fourierDimTrain)]
schwartz_3 = [60 * np.exp(-0.025 * (n ** 2)) for n in range(fourierDimTrain)]

schwartz_4 = [60 * np.exp(-0.01 * (n ** 2)) for n in range(fourierDimTest)]
schwartz_5 = [30 * np.exp(-0.01 * (n ** 2)) for n in range(fourierDimTest)]
schwartz_6 = [60 * np.exp(-0.025 * (n ** 2)) for n in range(fourierDimTest)]

if schwartz_train_function == 1:
    fourierRadiiTrain = schwartz_1
elif schwartz_train_function == 2:
    fourierRadiiTrain = schwartz_2
elif schwartz_train_function == 3:
    fourierRadiiTrain = schwartz_3

if schwartz_test_function == 1:
    fourierRadiiTest = schwartz_4
elif schwartz_test_function == 2:
    fourierRadiiTest = schwartz_5
elif schwartz_test_function == 3:
    fourierRadiiTest = schwartz_6

fourierRadiiTestMin = 0

if fourierDimTrain <= 5:
    min_train_radii = [ATLASradii[1] for n in range(fourierDimTrain)]
else: min_train_radii = [0 for n in range(fourierDimTrain)]

if fourierDimTest <= 5:
    min_test_radii = [ATLASradii[1] for n in range(fourierDimTest)]
elif Schwartz_train_test_on_disjoint_Sets == 'DISJOINT':
    min_test_radii = fourierRadiiTestMin
else: min_test_radii = [0 for n in range(fourierDimTest)]

# print("min test rad", min_test_radii)
# print("fourierRadiiTest", fourierRadiiTest)

fourierCenters = np.zeros(fourierDimTrain)
Lambda = np.max(ATLASradii)
min_dist_to_detector_layer = 0.05
times = np.linspace(0,Lambda, 500000)

plotting = False
plotting_datatype = 'train'
num_plotted_samples = 1
plotting_save_file = 'plot_4_of_tracks'
plot_title = "Schwarts space tracks"

chunk_size = 4


#if we wanted to make a different volume in fourier space, we could easily make a new function, ie. sample_from_cube or something analogous 
def sample_from_ball(chunk_size, max_radius, min_radius, center = np.zeros(3)):
    points = []
    while len(points) < chunk_size:
        vec = np.random.uniform(max_radius,-max_radius, 3)
        if (norm(vec) > max_radius) or (norm(vec) < min_radius):
            continue
        
        else:
            vec += center
            points.append(vec)

    return np.array(points)

def layerIDfunction(r):
    difference_array = np.absolute(ATLASradii - r)
    index = difference_array.argmin() + 1
    return index

def make_tracks_from_fourier_balls(chunk_size, fourierDim, radii, min_radii, centers):
    hyper_fourier_points = []
    for dimension in range(fourierDim):
        hyper_fourier_points.append(sample_from_ball(chunk_size,max_radius = radii[dimension], min_radius = min_radii[dimension],center=centers[dimension]))
    #print(len(hyper_fourier_points))
    return np.array(hyper_fourier_points)

def fourierExpand(fourierDim, Lambda, t, chunk_size = chunk_size):
    fourList = []
    shift = np.repeat(np.random.uniform(0, 2 * np.pi, (fourierDim, 3, chunk_size))[np.newaxis,:,:,:], len(t), axis = 0)
    #shift shape (time, fourDim, 3, chunk)
    for f_dimension in range(fourierDim):
        fourList.append(np.cos(2 * np.pi * f_dimension * t[:,np.newaxis, np.newaxis]/Lambda - shift[:,f_dimension,:,:]))
    fourList = np.array(fourList)
    return (fourList, shift)

import numpy as np

# Physical constants (exact)
_C = 2.99792458e8               # speed of light, m/s
_E_CHARGE = 1.602176634e-19     # elementary charge, C
# Conversion factor: 1 (GeV/c) -> kg*m/s  (use E/c where 1 GeV = 1.602176634e-10 J)
_GEV_C_TO_KGMS = (1.602176634e-10) / _C   # ≈ 5.344286e-19 kg·m/s

import numpy as np

def _count_layer_crossings_for_track(r_samples, z_samples, layer_radii_cm, z_limit_cm, tol_cm=0.5):
    """
    Count how many layer intersections a sampled track produces.
    Method: for each layer radius, check if r_samples crosses the radius (sign change)
    between consecutive sample points. Also require the z at crossing to be within +/- z_limit_cm/2.
    tol_cm: small tolerance applied to r difference when checking equality.
    """
    n_layers = len(layer_radii_cm)
    crossings = 0

    # r_samples, z_samples are 1D arrays over samples t
    for R_layer in layer_radii_cm:
        # evaluate f = r - R_layer
        f = r_samples - R_layer
        # find indices where sign change or very near crossing
        sc = np.where(np.abs(f) <= tol_cm)[0]
        if sc.size > 0:
            # if any sampled point already near the radius, check z at that point
            idx = sc[0]
            if abs(z_samples[idx]) <= z_limit_cm / 2.0:
                crossings += 1
                continue
        # else check sign changes between consecutive samples
        signs = f[:-1] * f[1:]
        sign_change_idx = np.where(signs < 0)[0]
        accepted = False
        for idx in sign_change_idx:
            # linear interpolation to approximate crossing z
            f1, f2 = f[idx], f[idx+1]
            if f2 == f1:
                frac = 0.5
            else:
                frac = abs(f1) / (abs(f1) + abs(f2))
            z_cross = z_samples[idx] + frac * (z_samples[idx+1] - z_samples[idx])
            if abs(z_cross) <= z_limit_cm / 2.0:
                crossings += 1
                accepted = True
                break
        # if accepted or found near-sample, we already counted
    return crossings

def tracks_standard_model_helix_cm_with_min_hits(
    t,
    chunk_size,
    radii,             # expects radii in centimeters (1D array)
    detector_length,   # centimeters
    B_field=1.0,       # Tesla
    min_hits=20,       # target minimal number of layer hits per track
    max_attempts=30,   # how many tries to accept a track before giving up
    phi_scale=1.0,     # radians per t-unit
    d0_max_cm=None,    # if None, will be set to ~0.3 * largest layer (helps crossings)
    R_choice='uniform_in_radius_cm',  # see below
    tol_cm=0.5,        # layer-crossing tolerance in cm
    random_seed=None,
):
    """
    Helix generator (returns ndarray shape (len(t), chunk_size, 3) in cm) that tries to
    ensure each track has at least `min_hits` layer intersections.

    Key behavior tuned to your dataset (which uses cm):
      - Instead of sampling pT directly, we sample R_cm uniformly across a useful range
        so tracks' transverse curvature spans the detector. Then compute pT accordingly.
      - d0_max_cm defaults to 0.3 * largest_layer if not provided (gives transverse offsets).
      - For each track we attempt up to max_attempts to sample parameters that yield >= min_hits.
        If max_attempts is reached we accept the best attempt.

    Parameters:
      - R_choice: 'uniform_in_radius_cm' (sample R directly), or 'derived_from_pT' (not used here).
      - returns tracks in centimeters (x_cm, y_cm, z_cm)
      
    Output: ndarray shape (len(t), chunk_size, 3) with coordinates in CM:
      (x_cm, y_cm, z_cm)

    Units / conventions:
      - pT, pz sampled in GeV/c (HEP momentum units).
      - B_field in Tesla.
      - radii and detector_length in centimeters (cm).
      - d0_max_cm in cm, outputs in cm.

    Physics -> unit conversion summary (explicit):
      Exact SI relation:
        R [m] = pT_SI [kg·m/s] / (e [C] * B [T])

      Convert pT from GeV/c to kg·m/s:
        1 GeV = 1.602176634e-10 J
        pT[kg·m/s] = pT_GeV * (1.602176634e-10) / c
                  = pT_GeV * (1.602176634e-10) / 2.99792458e8

      Combining constants gives the HEP shorthand:
        R [m] = pT[GeV/c] / (0.299792458 * B[T])   # exact numeric factor
      People often round 0.299792458 → 0.3 for quick estimates.

      To express R in CENTIMETERS:
        R [cm] = 100 * R [m] = 100 * pT / (0.299792458 * B)
               = pT / (0.00299792458 * B)

      This function uses the exact 0.299792458 and multiplies by 100 to produce R in cm.

    Parameter meanings:
      - phi = phi0 + q_sign * phi_scale * t
      - x = xc + R_cm * cos(phi)
      - y = yc + R_cm * sin(phi)
      - z = z0 + (pz/pT) * R_cm * (phi - phi0)   # all lengths in cm, pz/pT dimensionless

    Returns:
      ndarray (nt, chunk_size, 3) in cm
    """

    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random

    t = np.asarray(t)
    nt = len(t)
    layer_radii = np.asarray(radii)  # cm
    largest_layer = np.max(layer_radii)
    if d0_max_cm is None:
        d0_max_cm = max(1.0, 0.3 * largest_layer)  # heuristic: make centers offset enough

    tracks = np.zeros((nt, chunk_size, 3), dtype=float)

    # Precompute: convert desired R_cm range to pT if needed (here we work in R_cm directly)
    # Choose R_cm min/max so helices span inner->outer layers; use a slightly wider range
    R_min_cm = max( max(1.0, np.min(layer_radii)*0.5), 0.5 )      # at least ~1 cm
    R_max_cm = max( max( np.max(layer_radii)*1.2, R_min_cm + 1.0 ), 10.0 )

    best_candidates = [None] * chunk_size
    best_scores = [-1] * chunk_size  # best number of hits seen so far

    for track_idx in range(chunk_size):
        accepted = False
        best_track = None
        best_nhits = -1

        for attempt in range(max_attempts):
            # SAMPLE PARAMETERS FOR ONE TRACK
            q_sign = rng.choice([-1.0, 1.0])
            # sample transverse radius R in CM *directly* (guarantees sensible curvature)
            R_cm = rng.uniform(R_min_cm, R_max_cm)
            # pick phi0 and z0
            phi0 = rng.uniform(0.0, 2.0*np.pi)
            z0 = rng.uniform(-detector_length/2.0, detector_length/2.0)
            # center offset (impact parameter) in cm
            d0 = rng.uniform(0.0, d0_max_cm)
            d0_dir = rng.uniform(0.0, 2.0*np.pi)
            xc = d0 * np.cos(d0_dir)
            yc = d0 * np.sin(d0_dir)
            # pick pz/pT ratio by sampling a plausible pz and pT consistent with R_cm (for z advancement)
            # compute pT corresponding to R_cm: pT [GeV/c] = R_m * 0.299792458 * B
            R_m = R_cm * 0.01
            pT = R_m * 0.299792458 * B_field
            # make pT not vanishingly small; add small jitter
            pT = max(0.01, pT) * (1.0 + rng.uniform(-0.2, 0.2))
            # choose pz so pz/pT between -2 and 2 (controls pitch)
            pz = pT * rng.uniform(-1.5, 1.5)

            # build phi samples (nt,)
            phi = phi0 + (q_sign * phi_scale) * t  # shape (nt,)
            cosphi = np.cos(phi)
            sinphi = np.sin(phi)

            x = xc + R_cm * cosphi
            y = yc + R_cm * sinphi
            delta_phi = phi - phi0
            # z advance: z = z0 + (pz/pT) * R_cm * delta_phi
            z = z0 + ( (pz / pT) * R_cm * delta_phi )

            r = np.sqrt(x*x + y*y)

            # count how many distinct layer radii are crossed
            nhits = _count_layer_crossings_for_track(r, z, layer_radii, detector_length, tol_cm=tol_cm)

            # Keep best seen
            if nhits > best_nhits:
                best_nhits = nhits
                best_track = (x.copy(), y.copy(), z.copy())

            # Accept if meets threshold
            if nhits >= min_hits:
                accepted = True
                tracks[:, track_idx, 0] = x
                tracks[:, track_idx, 1] = y
                tracks[:, track_idx, 2] = z
                break

        # finished attempts for this track
        if not accepted:
            # fallback: accept best candidate found (may have fewer than min_hits)
            if best_track is None:
                # extremely unlikely: fallback to trivial straight small-radius
                phi = phi0 + (q_sign * phi_scale) * t
                x = xc + R_cm * np.cos(phi)
                y = yc + R_cm * np.sin(phi)
                z = z0 + ( (pz / pT) * R_cm * (phi - phi0) )
                tracks[:, track_idx, 0] = x
                tracks[:, track_idx, 1] = y
                tracks[:, track_idx, 2] = z
            else:
                tracks[:, track_idx, 0] = best_track[0]
                tracks[:, track_idx, 1] = best_track[1]
                tracks[:, track_idx, 2] = best_track[2]

        # Debug book-keeping optional: store best_scores
        best_scores[track_idx] = best_nhits

    # Optionally you can return (tracks, best_scores) to inspect how many hits each track has
    return (tracks, best_scores)  # in cm

def tracks_cylindrical_fourier_balls(t,fourierDim, Lambda, chunk_size, radii, min_radii, centers):

    #tracemalloc.start()

    fourierExp ,phase_shifts = fourierExpand(fourierDim, Lambda, t, chunk_size)
    cosPhases = np.cos(phase_shifts)
    fourierCoefficients = make_tracks_from_fourier_balls(chunk_size,fourierDim, radii, min_radii, centers) #(fourierDim,chunk_size ,3)
    print(fourierCoefficients)
    translate_to_origin = -np.sum(cosPhases[0] * np.transpose(fourierCoefficients, axes = [0,2,1]), axis = 0) # shapes (time, fourDim, 3, chunk),  (fourierDim,chunk_size ,3)
    cartesian_curve = np.sum(fourierExp * np.transpose(fourierCoefficients, axes=[0,2,1])[:,np.newaxis,:,:],axis = 0) # shapes (fourDim, time, 3, chunk), (fourierDim,chunk_size ,3), sum over fourierDim
    cartesian_curve = cartesian_curve + translate_to_origin
    #cartesian_curve has shape (time steps, coordinates, chunk_size)
    x = cartesian_curve[:,0,:]
    y = cartesian_curve[:,1,:]
    z = cartesian_curve[:,2,:]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)

    # current, peak = tracemalloc.get_traced_memory()

    # print(f"Current memory usage: {current / 10**6} MB")
    # print(f"Peak memory usage: {peak / 10**6} MB")

    # # Stop tracking memory allocations
    # tracemalloc.stop()


    return np.array(np.transpose(np.array([r,phi,z]),axes = [1,2,0]))

def map_curve_to_hits(curve, min_dist_to_detector_layer):

    #delete points not close enough to any detector layer
    close_points_index = np.argwhere((np.abs((curve[:,0])[:, np.newaxis]-ATLASradii[np.newaxis, :])).min(axis = 1) < min_dist_to_detector_layer).flatten()
    curve = curve[close_points_index, :]

    #group points based on which layer they are closest to
    closest_layer_per_point = np.argmin(np.abs((curve[:,0])[np.newaxis,:] - ATLASradii[:,np.newaxis]), axis = 0)

    #make lists of -1 and 1 to represent when the curve crosses a layer
    sign_list = np.sign(curve[:,0] - ATLASradii[closest_layer_per_point])

    #multiply by index of appropriate layer. We'll get list like (-1, -1, -1, 1, 1, 1 , -2, -2, 2, 2, -3 , -3, ...)
    sign_list = sign_list * (closest_layer_per_point + np.ones_like(closest_layer_per_point))

    hit_indices = np.argwhere(np.abs(np.roll(sign_list, 1) + sign_list) < 1/2)
    hit_indices = hit_indices[(hit_indices > 0) & (hit_indices < len(curve))]
    hits = curve[hit_indices, :]

    #append the layer id to the hits
    layerID = ((closest_layer_per_point + np.ones_like(closest_layer_per_point))[hit_indices])
    hits = np.concatenate((hits, (layerID[np.newaxis, :]).T), axis = 1)
    return hits

def make_list_of_hits_from_fourier_balls(chunk_size, Radii, min_radii, fourierDim, times, Centers, Lambda = np.max(ATLASradii), min_dist_to_detector_layer = 0.001):
    #Tracks has shape (chunk_size_train,fourDimTrain)
    Tracks = None
    if STANDARD_MODEL:
        Tracks_xyz, best_scores = tracks_standard_model_helix_cm_with_min_hits(t=times, chunk_size=chunk_size, radii=ATLASradii, detector_length=detector_length, B_field=bField, random_seed=42)
        x = Tracks_xyz[:, :, 0]
        y = Tracks_xyz[:, :, 1]
        z = Tracks_xyz[:, :, 2]
        r = np.sqrt(x*x + y*y)
        phi = np.arctan2(y, x)
        Tracks = np.stack([r, phi, z], axis=2)
    else:
        Tracks = tracks_cylindrical_fourier_balls(times,fourierDim,Lambda,chunk_size,Radii, min_radii, Centers)
    signal_hits = [] 
    for track in range(chunk_size):

        #curve has shape (time steps, coordinates)
        curve = Tracks[:,track,:]

        if plotting == True:
            intersection_points_df = pd.DataFrame(curve, columns = ['r', 'phi','z'])
            make_track_plot([intersection_points_df], 1)

        intersection_points = map_curve_to_hits(curve, min_dist_to_detector_layer)
        
        if ADD_HIT_NOISE and len(intersection_points) > 0:
            # intersection_points columns: r, phi, z, layer_id
            r0   = intersection_points[:, 0].astype(float)
            phi0 = intersection_points[:, 1].astype(float)
            z0   = intersection_points[:, 2].astype(float)

            x0 = r0 * np.cos(phi0)
            y0 = r0 * np.sin(phi0)

            # Add Gaussian measurement noise
            x1 = x0 + np.random.normal(0.0, HIT_NOISE_SIGMA_XY, size=len(x0))
            y1 = y0 + np.random.normal(0.0, HIT_NOISE_SIGMA_XY, size=len(y0))
            z1 = z0 + np.random.normal(0.0, HIT_NOISE_SIGMA_Z,  size=len(z0))

            # Convert back to cylindrical
            r1   = np.sqrt(x1**2 + y1**2)
            phi1 = np.arctan2(y1, x1)

            intersection_points[:, 0] = r1
            intersection_points[:, 1] = phi1
            intersection_points[:, 2] = z1

        intersection_points = np.hstack((((track + 1) * np.ones(len(intersection_points))[np.newaxis, :]).T, intersection_points))

        intersection_points_df = pd.DataFrame(intersection_points, columns = ['particle_id','r', 'phi','z', 'layer_id'])
        
        if WRITE_HIT_SIGMAS:
            # Constant σ in x/y; translate to σ_r and σ_phi approximately for convenience
            # σ_r ~ σ_xy (small-angle approximation)
            intersection_points_df["sigma_r"] = HIT_NOISE_SIGMA_XY
            intersection_points_df["sigma_z"] = HIT_NOISE_SIGMA_Z

            # σ_phi ≈ σ_xy / r (avoid divide-by-zero)
            r_safe = np.maximum(intersection_points_df["r"].to_numpy(), 1e-6)
            intersection_points_df["sigma_phi"] = HIT_NOISE_SIGMA_XY / r_safe

        #Delete everything past when the particle leaves the detector

        intersection_points_df = intersection_points_df.loc[:intersection_points_df[(intersection_points_df['r'] >= max_B_radius)].index.min()]
        intersection_points_df = intersection_points_df.loc[:intersection_points_df[np.abs(intersection_points_df['z']) >= detector_length/2].index.min()]
        signal_hits.append(intersection_points_df)
    return signal_hits

def make_track_plot(tracks, num_plotted_samples, show = False):
    # TracksTrain has shape (chunk_size_train, fourDimTrain)
    
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Plot all tracks
    for track in range(num_plotted_samples):
        # curve has shape (time steps, coordinates)
        
        curve = tracks[track]
        #print(curve)
        r = curve['r']
        phi = curve['phi']
        z = curve['z']

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        #print("making plot!!")
        # Add each track to the same 3D plot
        ax.scatter3D(x, y, z, s=3, label=f"Track {track}")
        break

    # Create concentric cylinders
    theta = np.linspace(0, 2 * np.pi, 100)  # Angle for the circular base of the cylinder
    z_range = np.linspace(np.min([np.min(df['z']) for df in tracks]), np.max([np.max(df['z']) for df in tracks]), 100)  # z-axis range

    for radius in ATLASradii[22:]:
        # Meshgrid for the cylinder
        Theta, Z = np.meshgrid(theta, z_range)
        X = radius * np.cos(Theta)
        Y = radius * np.sin(Theta)
        
        # Plot each cylinder with semi-transparency
        ax.plot_surface(X, Y, Z, color='cyan', alpha=0.2, rstride=5, cstride=5)

    # Final plot settings
    plt.title(plot_title)
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(plotting_save_file)


def prepare_signal_dfs(chunk, chunk_size, fourierRadii, min_radii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                       event_id_minus_event, final_iteration = False, signal_hits = None, remaining_events_after_chunks = None):
    if final_iteration == False:
        signal_hits = make_list_of_hits_from_fourier_balls(chunk_size, fourierRadii, min_radii, fourierDim,times , fourierCenters,Lambda , 
                                                           min_dist_to_detector_layer)
        range_for_events = chunk_size
    elif final_iteration == True:
        range_for_events = remaining_events_after_chunks
    if (num_plotted_samples > chunk_size) & (plotting == True):
        raise ValueError("num_plotted_samples must be smaller than chunk_size")
    if (plotting == True) & (chunk == 0):
        make_track_plot(signal_hits, num_plotted_samples)

    for item in range(range_for_events):
        if final_iteration == False:
            event_id = event_id_minus_event + item
        elif final_iteration == True:
            event_id = event_id_minus_event
        signal_hits_df = signal_hits[item]

        signal_particle_df = pd.DataFrame({
            'particle_id':[1],
            'vx':[10],
            'vy':[10],
            'vz':[10],
            'px':[10],
            'py':[10],
            'pz':[10],
            'charge':[1],
            'PID':[signal_PID],
            'Event#':[event_id],
            'Mass':[1000]
        })
        signal_hits_df['particle_id'] = 1
        signal_hits_df['hit_id'] = signal_hits_df.index + 1
        signal_hits_df['layer_id'] = signal_hits_df['layer_id'].astype(int)
        signal_particle_df.to_csv(os.path.join(output_dir,new_output_folder,f'event{event_id + 100000000}-particles.csv'), index=False) 
        signal_hits_df.to_csv(os.path.join(output_dir,new_output_folder,f'event{event_id + 100000000}-hits.csv'), index=False)
        del signal_particle_df
        del signal_hits_df
        

def make_files(datatype, signal_tracks_per_event, fourierRadii,fourierDim ,times, fourierCenters, min_radii, Lambda = np.max(ATLASradii),
               min_dist_to_detector_layer = 0.0001, data_combination = 'SM and Signal'):
    if os.path.exists(os.path.join(output_dir, new_output_folder)):
        pass
    else:
        os.mkdir(os.path.join(output_dir, new_output_folder))
    
    dataset_dir = os.path.join(output_dir, new_output_folder)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # Write manifest once
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        manifest = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "NUM_TRAIN_TRACKS": train_size,
            "NUM_TEST_TRACKS": test_size,
            "NUMBER_OF_LAYERS": num_layers,
            "DETECTOR_LENGTH": detector_length,
            "SMALLEST_LAYER": smallest_layer,
            "LARGEST_LAYER": largest_layer,
            "FOURIER_DIM_TRAIN": fourierDimTrain - 1,  # because script adds 1 later
            "FOURIER_DIM_TEST": fourierDimTest - 1,
            "TRAIN_FUNCTION": schwartz_train_function,
            "TEST_FUNCTION": schwartz_test_function,
            "DISJOINT": Schwartz_train_test_on_disjoint_Sets,
            "ADD_HIT_NOISE": ADD_HIT_NOISE,
            "HIT_NOISE_SIGMA_XY": HIT_NOISE_SIGMA_XY,
            "HIT_NOISE_SIGMA_Z": HIT_NOISE_SIGMA_Z,
            "WRITE_HIT_SIGMAS": WRITE_HIT_SIGMAS,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    if datatype == 'train':
        datatype_size = train_size
    elif datatype == 'validate':
        datatype_size = validate_size
    elif datatype == 'test':
        datatype_size = test_size

    number_of_chunks = np.floor(datatype_size/chunk_size).astype(int)
    remaining_events_after_chunks = datatype_size % chunk_size
    
    for chunk in range(number_of_chunks):
        # event is used as the iterator within the loop inside combine_SM_and_signal_dfs, so we pass event_id_minus_event into the function and add event to it with each loop iteration
        if datatype == 'train':
            event_id_minus_event = chunk * chunk_size + 1
        elif datatype == 'validate':
            event_id_minus_event = train_size + chunk * chunk_size + 1
        elif datatype == 'test':
            event_id_minus_event = train_size + validate_size + chunk * chunk_size + 1
        

        prepare_signal_dfs(chunk, chunk_size, fourierRadii, min_radii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                            event_id_minus_event, final_iteration = False, signal_hits = None, remaining_events_after_chunks = None)
            
    if remaining_events_after_chunks > 0:
        signal_hits = make_list_of_hits_from_fourier_balls(remaining_events_after_chunks, fourierRadii, min_radii, fourierDim,times , 
                                                        fourierCenters,Lambda , min_dist_to_detector_layer)
        for event in range(remaining_events_after_chunks):
            if datatype == 'train':
                event_id = event + number_of_chunks * chunk_size + 1
            elif datatype == 'validate':
                event_id = event + train_size + number_of_chunks * chunk_size + 1
            elif datatype == 'test':
                event_id = event + train_size + validate_size + number_of_chunks * chunk_size + 1


            prepare_signal_dfs(chunk, chunk_size, fourierRadii, min_radii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                            event_id, final_iteration = True, signal_hits = signal_hits, 
                            remaining_events_after_chunks = remaining_events_after_chunks)

if __name__ == '__main__':

    print("making train set")
    make_files(datatype = 'train', signal_tracks_per_event = signal_tracks_per_event,fourierRadii = fourierRadiiTrain,fourierDim = fourierDimTrain,
              times = times, fourierCenters = fourierCenters, min_radii=min_train_radii,Lambda = Lambda, min_dist_to_detector_layer = min_dist_to_detector_layer, data_combination='signal')

    print("making test set")
    make_files(datatype = 'test', signal_tracks_per_event = signal_tracks_per_event,fourierRadii = fourierRadiiTest,fourierDim = fourierDimTest,
              times = times, fourierCenters = fourierCenters, min_radii=min_test_radii, Lambda = Lambda, min_dist_to_detector_layer = min_dist_to_detector_layer, data_combination ='signal')


