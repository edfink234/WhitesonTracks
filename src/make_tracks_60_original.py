import os
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt

#user defined input
num_layers = os.getenv('NUMBER_OF_LAYERS', 25)
detector_length = os.getenv('DETECTOR_LENGTH', 320)
smallest_layer = os.getenv('SMALLEST_LAYER', 3.1)
largest_layer = os.getenv('LARGEST_LAYER', 53)
new_output_folder = os.getenv('OUTPUT_FOLDER')
fourierDimTest = os.getenv('FOURIER_DIM_TEST', 25)
fourierDimTrain = os.getenv('FOURIER_DIM_TRAIN', 25)
train_size = os.getenv('NUM_TRAIN_TRACKS')
test_size = os.getenv('NUM_TEST_TRACKS')
schwartz_train_function = os.getenv('TRAIN_FUNCTION', 3)
schwartz_test_function = os.getenv('TEST_FUNCTION', 6)

ATLASradii = np.linspace(smallest_layer, largest_layer, num_layers)


validate_size = 0

fourierDimValidate = 1

intersection_of_sets = 4
#new_output_folder = "test_6"
signal_PID = 15
bField = 1
max_B_radius = np.max(ATLASradii)
Schwartz_train_test_on_disjoint_Sets = os.getenv('DISJOINT', False)
start_train_dim = 0
start_test_dim = 4
input_dir = '/pscratch/sd/l/lcondren/Event_files_93000'
output_dir = '/pscratch/sd/l/lcondren/combined_hit_particle_files'


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

        intersection_points = np.hstack((((track + 1) * np.ones(len(intersection_points))[np.newaxis, :]).T, intersection_points))

        intersection_points_df = pd.DataFrame(intersection_points, columns = ['particle_id','r', 'phi','z', 'layer_id'])

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
        

def make_files(input_dir, datatype, signal_tracks_per_event, fourierRadii,fourierDim ,times, fourierCenters, min_radii,Lambda = np.max(ATLASradii), 
               min_dist_to_detector_layer = 0.0001, data_combination = 'SM and Signal'):
    if os.path.exists(os.path.join(output_dir, new_output_folder)):
        pass
    else:
        os.mkdir(os.path.join(output_dir, new_output_folder))
    
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
make_files(input_dir = input_dir, datatype = 'train', signal_tracks_per_event = signal_tracks_per_event,fourierRadii = fourierRadiiTrain,fourierDim = fourierDimTrain, 
          times = times, fourierCenters = fourierCenters, min_radii=min_train_radii,Lambda = Lambda, min_dist_to_detector_layer = min_dist_to_detector_layer, data_combination='signal')

print("making test set")
make_files(input_dir = input_dir, datatype = 'test', signal_tracks_per_event = signal_tracks_per_event,fourierRadii = fourierRadiiTest,fourierDim = fourierDimTest, 
          times = times, fourierCenters = fourierCenters, min_radii=min_test_radii, Lambda = Lambda, min_dist_to_detector_layer = min_dist_to_detector_layer, data_combination ='signal')
