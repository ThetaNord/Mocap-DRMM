''' Loads the CMU Mocap data from a zipfile, preprocesses it according to
    specifications, creates a motion graph dataset and saves it as a numpy file.
'''
import os, sys, argparse
import pandas as pd
import numpy as np
import zipfile

from sklearn.model_selection import train_test_split

# General parameters
DEBUG_LEVEL = 1

# Data parameters
CSV_COLUMNS = ['is_first', 'hips', 'spine', 'left_upper_leg', 'left_lower_leg',
    'left_foot', 'right_upper_leg', 'right_lower_leg', 'right_foot', 'left_shoulder',
    'left_upper_arm', 'left_lower_arm', 'left_hand', 'left_toes', 'right_toes',
    'right_shoulder', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'head', 'neck', 'none']
CSV_COLUMNS_REORDERED = ['left_hand', 'right_hand', 'left_lower_arm', 'right_lower_arm',
    'left_upper_arm', 'right_upper_arm', 'left_shoulder', 'right_shoulder', 'head', 'neck',
    'spine', 'hips', 'left_upper_leg', 'right_upper_leg', 'left_lower_leg', 'right_lower_leg',
    'left_foot', 'right_foot', 'left_toes', 'right_toes']
CSV_COLUMNS_MIRRORED = ['right_hand', 'left_hand', 'right_lower_arm','left_lower_arm',
    'right_upper_arm', 'left_upper_arm', 'right_shoulder', 'left_shoulder', 'head', 'neck',
    'spine', 'hips', 'right_upper_leg', 'left_upper_leg', 'right_lower_leg', 'left_lower_leg',
    'right_foot', 'left_foot', 'right_toes', 'left_toes']
ROOT_COLUMN = 'hips'         # column label for the body part to use as root
DROP_COLUMNS = ['is_first', 'none']
SHOULDER_COLUMNS = ['left_shoulder', 'right_shoulder']
CSV_DELIMITER = ';'

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        dest='data_path',
        help='relative path to the zip file containing the data to process',
        default='data/cmu_tiny.zip',
        type=str
    )
    parser.add_argument(
        '--output-path',
        dest='output_path',
        help='relative path to save the output',
        default='data/motion_graph_test.npz',
        type=str
    )
    parser.add_argument(
        '--seq-length',
        dest='sequence_length',
        help='how many time steps per sequence',
        default=64,
        type=int
    )
    parser.add_argument(
        '--dataset-size',
        dest='dataset_size',
        help='how many animations to create for the dataset',
        default=10,
        type=int
    )
    parser.add_argument(
        '--test-size',
        dest='test_set_size',
        help='how large a portion of the data to separate into test set',
        default=0.1,
        type=float
    )
    parser.add_argument(
        '--mirror-animations',
        dest='mirror_animations',
        help='augment data by left-right mirroring animations',
        action='store_true'
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        help='seed for initializing tensorflow random number generation',
        default=0,
        type=int
    )
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

# Takes a numpy array of shape [N, 3M] and rotates each [1,3] subset based on the given sine and cosine
def rotateMatrix(matrix, cosine, sine):
    if DEBUG_LEVEL > 1: print("Original matrix:\n{}".format(matrix))
    cos_matrix = np.resize(np.array([cosine, 1, cosine]), matrix.shape)
    if DEBUG_LEVEL > 1: print("Cosine matrix:\n{}".format(cos_matrix))
    rot_matrix = np.multiply(cos_matrix, matrix)
    if DEBUG_LEVEL > 1: print("Intermediate matrix:\n{}".format(rot_matrix))
    aux_matrix = np.zeros(matrix.shape)
    aux_matrix[:, 0::3] = -sine * np.copy(matrix[:, 2::3])
    aux_matrix[:, 2::3] = sine * np.copy(matrix[:, 0::3])
    if DEBUG_LEVEL > 1: print("Auxiliary matrix:\n{}".format(aux_matrix))
    rot_matrix = np.add(rot_matrix, aux_matrix)
    if DEBUG_LEVEL > 1:
        print("Rotated matrix:\n{}".format(rot_matrix))
        print("Difference:\n{}".format(np.subtract(rot_matrix, matrix)))
    return rot_matrix

# Normalize position and rotation of a sequence of frames based on first frame
# Note: The size of the frames array is not restricted
def normalize_frames(frames):
    # Determine root node and shoulder indices for normalization
    root_index = CSV_COLUMNS_REORDERED.index(ROOT_COLUMN)*3
    shoulder_indices = 3 * np.array([CSV_COLUMNS_REORDERED.index(SHOULDER_COLUMNS[0]),
        CSV_COLUMNS_REORDERED.index(SHOULDER_COLUMNS[1])])
    # Normalize position based on root position of first frame
    origin = frames[0, root_index:root_index+3].copy()
    origin[1] = 0.0
    normalized_frames = frames - np.resize(origin, frames.shape)
    # Normalize rotation (figure faces along z axis) based on first frame
    shoulder_vector = normalized_frames[0, shoulder_indices[1]:shoulder_indices[1]+3] - normalized_frames[0, shoulder_indices[0]:shoulder_indices[0]+3]
    shoulder_vector[1] = 0
    axis = np.array([0,0,1])
    # Calculate sine and cosine of angle between vector and z axis (in radians)
    magnitude = np.linalg.norm(shoulder_vector)
    cosine = shoulder_vector.dot(axis)/magnitude
    sine = np.linalg.norm(np.cross(shoulder_vector, axis))/magnitude
    normalized_frames = rotateMatrix(normalized_frames, cosine, np.sign(shoulder_vector[0])*sine)
    # Return normalized frames
    return normalized_frames

def processDataFrame(dataFrame, mirror=False):
    # Drop unused columns
    for column in DROP_COLUMNS:
        if column in dataFrame:
            dataFrame = dataFrame.drop(column, axis=1)
    if mirror: dataFrame = dataFrame[CSV_COLUMNS_MIRRORED]
    else: dataFrame = dataFrame[CSV_COLUMNS_REORDERED]
    # Split all columns to six parts
    processedFrame = dataFrame.stack().str.extractall('([\d\.E-]+)').unstack([-2, -1])
    # Drop columns that contain rotations
    processedFrame.columns = np.arange(len(processedFrame.columns))
    drop_indices = np.arange(processedFrame.count(axis=1)[0])
    drop_indices = np.where(drop_indices % 6 > 2)
    drop_indices = drop_indices[0].tolist()
    processedFrame = processedFrame.drop(drop_indices, axis='columns')
    # Convert all columnts to float
    processedFrame = processedFrame.astype(np.float64)
    # Convert from data frame to numpy array
    processedFrame = processedFrame.values
    # Normalize frames
    processedFrame = normalize_frames(processedFrame)
    # Mirror frames, if required
    if mirror:
        # Flip the coordinates relative to the z-axis
        multiplier = np.resize(np.array([1, 1, -1]), processedFrame.shape)
        processedFrame = np.multiply(processedFrame, multiplier)
    # Return the processed frame
    return processedFrame

def file_to_sequences(zf, filename, sequence_length, step_size, mirror_animations):
    # Load the contents of the file into a dataframe
    data_frame = pd.read_csv(zf.open(filename), sep=CSV_DELIMITER, names=CSV_COLUMNS)
    # Initiate list for storing sequences
    sequences = []
    # Skip the file if the sequence is too short overall
    animation_length = data_frame.count(axis=0)[0]
    if animation_length < sequence_length:
        if DEBUG_LEVEL > 0: print("Animation too short. Skipping...")
        return sequences
    # Split dataframe into multiple sequences and process individually
    start = 0
    end = sequence_length-1
    while end < animation_length:
        current_frame = data_frame.copy().truncate(before=start, after=end)
        # Reindex rows
        current_frame.index = range(len(current_frame.index))
        data_array = processDataFrame(current_frame, mirror=False)
        # Store the dataframe in the list
        sequences.append(data_array)
        if mirror_animations:
            mirrored_array = processDataFrame(current_frame, mirror=True)
            sequences.append(mirrored_array)
        start += step_size
        end += step_size
    return sequences

# Calculate limits for maximum changes in joints between frames
def calculate_limits(sequences):
    max_speeds = np.zeros(len(CSV_COLUMNS_REORDERED))
    max_accelerations = np.zeros(len(CSV_COLUMNS_REORDERED))
    max_jerks = np.zeros(len(CSV_COLUMNS_REORDERED))
    # Calculate velocity and acceleration for each joint in each frame
    for k in range(sequences.shape[0]):
        sequence = sequences[k]
        sequence_velocities = []
        sequence_accelerations = []
        for i in range(1, sequence.shape[0]):
            # Calculate velocities
            velocities = np.zeros((len(CSV_COLUMNS_REORDERED), 3))
            for j in range(len(CSV_COLUMNS_REORDERED)):
                joint_start = sequence[i-1, 3*j:3*(j+1)]
                joint_end = sequence[i, 3*j:3*(j+1)]
                velocity = joint_end - joint_start
                speed = np.linalg.norm(velocity)
                velocities[j] = velocity
                if max_speeds[j] < speed:
                    max_speeds[j] = speed
            # Calculate accelerations
            if i >= 2:
                prev_velocities = sequence_velocities[-1]
                accelerations = velocities - prev_velocities
                for j in range(accelerations.shape[0]):
                    scalar_acceleration = np.linalg.norm(accelerations[j])
                    if max_accelerations[j] < scalar_acceleration:
                        max_accelerations[j] = scalar_acceleration
                # Calculate jerks
                if i >= 3:
                    prev_accelerations = sequence_accelerations[-1]
                    jerks = accelerations - prev_accelerations
                    for j in range(jerks.shape[0]):
                        scalar_jerk = np.linalg.norm(jerks[j])
                        if max_jerks[j] < scalar_jerk:
                            max_jerks[j] = scalar_jerk
                sequence_accelerations.append(accelerations)
            sequence_velocities.append(velocities)
    limits = {"speed_limits": max_speeds, "acceleration_limits": max_accelerations, "jerk_limits": max_jerks}
    return limits

# Verify that adding new_frame to animation would not break given limits
def verify_frame(animation, new_frame, limits):
    if animation != None and len(animation) > 0:
        for j in range(len(CSV_COLUMNS_REORDERED)):
            joint_start = animation[-1][3*j:3*(j+1)]
            joint_end = new_frame[3*j:3*(j+1)]
            velocity = joint_end - joint_start
            speed = np.linalg.norm(velocity)
            if speed > limits["speed_limits"][j]:
                return False
            if len(animation) > 1:
                prev_velocity = animation[-1][3*j:3*(j+1)] - animation[-2][3*j:3*(j+1)]
                acceleration = velocity - prev_velocity
                scalar_acceleration = np.linalg.norm(acceleration)
                if scalar_acceleration > limits["acceleration_limits"][j]:
                    return False
                if len(animation) > 2:
                    prev_acceleration = prev_velocity - (animation[-2][3*j:3*(j+1)] - animation[-3][3*j:3*(j+1)])
                    jerk = acceleration - prev_acceleration
                    scalar_jerk = np.linalg.norm(jerk)
                    if scalar_jerk > limits["jerk_limits"][j]:
                        return False
    return True

def create_animation(frames, animation_length, limits, always_return=False):
    # Initialize a list for storing the animation
    animation_sequence = []
    root_index = CSV_COLUMNS_REORDERED.index(ROOT_COLUMN)*3
    # Index all frames
    frame_index = list(range(len(frames)))
    # Randomly pick initial frame and add it to animation
    idx = np.random.choice(frame_index)
    animation_sequence.append(frames[idx])
    # Remove selected frame from index
    frame_index.remove(idx)
    temp_index = frame_index.copy()
    # Keep picking additional frames, adhering to limits, until animation is the right length or index is exhausted
    while len(animation_sequence) < animation_length and len(temp_index) > 0:
        idx = np.random.choice(temp_index)
        frame = frames[idx]
        origin = animation_sequence[-1][root_index:root_index+3].copy()
        origin[1] = 0.0
        frame = frame + np.resize(origin, frame.shape)
        if verify_frame(animation_sequence, frame, limits):
            #print("Selected index: {}".format(idx))
            animation_sequence.append(frame)
            frame_index.remove(idx)
            temp_index = frame_index.copy()
        else:
            temp_index.remove(idx)
    # Convert animation sequence into a numpy array
    animation = None
    if always_return or len(animation_sequence) == animation_length:
        animation = np.array(animation_sequence)
        # Normalize animation
        animation = normalize_frames(animation)
        print("Animation completed!")
    else:
        print("Animation unfinished...")
    return animation

def create_animation_dataset(zf, files, args):
    # Initialize a list for storing the data
    dataset = []
    # Part 1: Get full sequences from files
    sequences = []
    # Loop over each file in archive
    for f in files:
        # Load the contents of the file into a dataframe
        data_frame = pd.read_csv(zf.open(f.filename), sep=CSV_DELIMITER, names=CSV_COLUMNS)
        # Process data frame
        sequence = processDataFrame(data_frame, mirror=False)
        sequences.append(sequence)
    sequences = np.array(sequences)
    print("Sequences loaded")
    # Part 2: Calculate limits
    limits = calculate_limits(sequences)
    print("Limits calculated")
    # Part 3: Get individual frames
    sequence_list = []
    # Loop over all the files in the archive
    for f in files:
        # Verify that the file is a CSV file
        if f.filename.endswith(".csv"):
            sequences = file_to_sequences(zf, f.filename, 2, 1, args.mirror_animations)
            sequence_list.extend(sequences)
    full_data = np.array(sequence_list)
    frames = full_data[:, 1]
    print("Frames collected")
    print("Frames shape: {}".format(frames.shape))
    # Part 4: Create a list of animations
    while len(dataset) < args.dataset_size:
        new_animation = create_animation(frames, args.sequence_length, limits)
        if new_animation is not None:
            dataset.append(new_animation)
    print("Animations created")
    # Part 5: Split the dataset into train, validation and test sets
    dataset = np.array(dataset)
    train_data, test_data = train_test_split(dataset, test_size=args.test_set_size, random_state=args.seed)
    train_data, validation_data = train_test_split(train_data, test_size=test_data.shape[0], random_state=args.seed)
    print("Dataset split complete")
    return train_data, validation_data, test_data

def main(args):
    # Load the zip file from the given path
    zf = zipfile.ZipFile(args.data_path)
    # Get list of files contained within the archive
    files = zf.infolist()
    # Create dataset arrays
    train_data, validation_data, test_data = create_animation_dataset(zf, files, args)
    print("Train dataset size: {}".format(train_data.shape[0]))
    print("Validation dataset size: {}".format(validation_data.shape[0]))
    print("Test dataset size: {}".format(test_data.shape[0]))
    # Save the arrays into a npz file
    np.savez(args.output_path, train_data=train_data, validation_data=validation_data, test_data=test_data)

if __name__ == '__main__':
    # Parse command line arguments
    argv = sys.argv
    args = parse_args(argv)
    if DEBUG_LEVEL > 1:
        print(args)
    main(args)
