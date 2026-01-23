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
    "NUM_TRAIN_TRACKS": 10,
    "NUM_TEST_TRACKS": 10,

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
}

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

def make_list_of_hits_from_fourier_balls(chunk_size, Radii, min_radii,fourierDim,times ,Centers ,Lambda = np.max(ATLASradii), min_dist_to_detector_layer = 0.001):
    #Tracks has shape (chunk_size_train,fourDimTrain)

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

def make_track_plot(tracks, num_plotted_samples):
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
            
print("making train set")
make_files(datatype = 'train', signal_tracks_per_event = signal_tracks_per_event,fourierRadii = fourierRadiiTrain,fourierDim = fourierDimTrain,
          times = times, fourierCenters = fourierCenters, min_radii=min_train_radii,Lambda = Lambda, min_dist_to_detector_layer = min_dist_to_detector_layer, data_combination='signal')

print("making test set")
make_files(datatype = 'test', signal_tracks_per_event = signal_tracks_per_event,fourierRadii = fourierRadiiTest,fourierDim = fourierDimTest,
          times = times, fourierCenters = fourierCenters, min_radii=min_test_radii, Lambda = Lambda, min_dist_to_detector_layer = min_dist_to_detector_layer, data_combination ='signal')
