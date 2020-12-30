''' Loads the CMU Mocap data from a zipfile, preprocesses it according to
    specifications and saves it as a numpy file.
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

# Preprocessing Parameters
NORMALIZE_POSITION = True
NORMALIZE_ROTATION = True
#REORDER_COLUMNS = True

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        dest='data_path',
        help='relative path to the zip file containing the data to process',
        default='data/mocap-csv.zip',
        type=str
    )
    parser.add_argument(
        '--output-path',
        dest='output_path',
        help='relative path to save the output',
        default='data/mocap-dataset.npz',
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
        '--step-size',
        dest='step_size',
        help='how many time steps to shift the window between sequences',
        default=64,
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
        '--split-mode',
        dest='split_mode',
        help='whether to split the test set off first or last (or both)',
        default='last',
        type=str
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

def processDataFrame(dataFrame, mirror=False):
    # Drop unused columns
    for column in DROP_COLUMNS:
        if column in dataFrame:
            dataFrame = dataFrame.drop(column, axis=1)
    if DEBUG_LEVEL > 1: print("Columns before reorder:\n{}".format(dataFrame.columns))
    if mirror: dataFrame = dataFrame[CSV_COLUMNS_MIRRORED]
    else: dataFrame = dataFrame[CSV_COLUMNS_REORDERED]
    if DEBUG_LEVEL > 1: print("Columns after reorder:\n{}".format(dataFrame.columns))
    # Get root column index
    root_index = dataFrame.columns.get_loc(ROOT_COLUMN)*3
    # Get shoulder shoulder indices
    shoulder_indices = 3 * np.array([dataFrame.columns.get_loc(SHOULDER_COLUMNS[0]),
            dataFrame.columns.get_loc(SHOULDER_COLUMNS[1])])
    if DEBUG_LEVEL > 1: print(root_index)
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
    if NORMALIZE_POSITION:
        # Normalize position based on first frame
        if DEBUG_LEVEL > 1: print(processedFrame[-1])
        origin = processedFrame[0, root_index:root_index+3].copy()
        origin[1] = 0.0
        if DEBUG_LEVEL > 1: print(origin)
        normalizingArray = np.resize(origin, processedFrame.shape)
        processedFrame = processedFrame - normalizingArray
        if DEBUG_LEVEL > 1: print(processedFrame[-1])
    if NORMALIZE_ROTATION:
        if DEBUG_LEVEL > 1: print(processedFrame[0])
        # Normalize rotation (figure faces along z axis)
        shoulder_vector = processedFrame[0, shoulder_indices[1]:shoulder_indices[1]+3] - processedFrame[0, shoulder_indices[0]:shoulder_indices[0]+3]
        shoulder_vector[1] = 0
        if DEBUG_LEVEL > 1: print(shoulder_vector)
        axis = np.array([0,0,1])
        # Calculate sine and cosine of angle between vector and z axis (in radians)
        magnitude = np.linalg.norm(shoulder_vector)
        cosine = shoulder_vector.dot(axis)/magnitude
        sine = np.linalg.norm(np.cross(shoulder_vector, axis))/magnitude
        if DEBUG_LEVEL > 1:
            print("Shoulder vector: {}".format(shoulder_vector))
            print("Axis: {}".format(axis))
            print("Cosine before rotation: {}".format(cosine))
            print("Sine before rotation: {}".format(sine))
        processedFrame = rotateMatrix(processedFrame, cosine, np.sign(shoulder_vector[0])*sine)
        if DEBUG_LEVEL > 1:
            #print(processedFrame[0])
            shoulder_vector = processedFrame[0, shoulder_indices[1]:shoulder_indices[1]+3] - processedFrame[0, shoulder_indices[0]:shoulder_indices[0]+3]
            shoulder_vector[1] = 0
            magnitude = np.linalg.norm(shoulder_vector)
            cosine = shoulder_vector.dot(axis)/magnitude
            sine = np.linalg.norm(np.cross(shoulder_vector, axis))/magnitude
            print("Cosine after rotation: {}".format(cosine))
            print("Sine after rotation: {}".format(sine))
            radians = np.arccos(cosine)
            print("Radians after rotation (should be zero): {}".format(radians))
    if mirror:
        # Flip the coordinates relative to the z-axis
        multiplier = np.resize(np.array([1, 1, -1]), processedFrame.shape)
        processedFrame = np.multiply(processedFrame, multiplier)
    # Return the processed frame
    return processedFrame

def fileToSequences(zf, filename, sequence_length, step_size, mirror_animations):
    # Load the contents of the file into a dataframe
    dataFrame = pd.read_csv(zf.open(filename), sep=CSV_DELIMITER, names=CSV_COLUMNS)
    # Initiate list for storing sequences
    sequences = []
    # Skip the file if the sequence is too short overall
    animationLength = dataFrame.count(axis=0)[0]
    if animationLength < sequence_length:
        if DEBUG_LEVEL > 0: print("Animation too short. Skipping...")
        return sequences
    # Split dataframe into multiple sequences and process individually
    start = 0
    end = sequence_length-1
    while end < animationLength:
        currentFrame = dataFrame.copy().truncate(before=start, after=end)
        # Reindex rows
        currentFrame.index = range(len(currentFrame.index))
        if DEBUG_LEVEL > 1: print(currentFrame)
        dataArray = processDataFrame(currentFrame)
        if DEBUG_LEVEL > 1: print(dataArray)
        # Store the dataframe in the list
        sequences.append(dataArray)
        if mirror_animations:
            mirrored_array = processDataFrame(currentFrame, mirror=True)
            sequences.append(mirrored_array)
            if DEBUG_LEVEL > 1:
                mirror_sum = np.add(dataArray[:, 26:38:3], mirrored_array[:, 26:38:3])
                print(mirror_sum)
        start += step_size
        end += step_size
    return sequences

def dataFirstSplit(zf, files, args):
    train_list = []
    test_list = []
    train_files, test_files = train_test_split(np.array(files), test_size=args.test_set_size, random_state=args.seed)
    if DEBUG_LEVEL > 0:
        print("Train files: {}".format(train_files.shape[0]))
        print("Test files: {}".format(test_files.shape[0]))
    #train_count = 0
    #test_count = 0
    for f in train_files:
        # Make sure that the file is a CSV file
        if f.filename.endswith(".csv"):
            if DEBUG_LEVEL > 0: print(f.filename)
            sequences = fileToSequences(zf, f.filename, args.sequence_length, args.step_size, args.mirror_animations)
            #if len(sequences) > 0: train_count += 1
            train_list.extend(sequences)
    for f in test_files:
        # Make sure that the file is a CSV file
        if f.filename.endswith(".csv"):
            if DEBUG_LEVEL > 0: print(f.filename)
            sequences = fileToSequences(zf, f.filename, args.sequence_length, 64)
            #if len(sequences) > 0: test_count += 1
            test_list.extend(sequences)
    train_data = np.array(train_list)
    test_data = np.array(test_list)
    #print("Training clips: {}".format(train_count))
    #print("Test clips: {}".format(test_count))
    return train_data, test_data

def dataLastSplit(zf, files, args):
    # Initiate an empty list for storing the data
    data_list = []
    # Loop over all the files in the archive
    for f in files:
        # Make sure that the file is a CSV file
        if f.filename.endswith(".csv"):
            if (DEBUG_LEVEL > 0): print(f.filename)
            sequences = fileToSequences(zf, f.filename, args.sequence_length, args.step_size, args.mirror_animations)
            data_list.extend(sequences)
    if DEBUG_LEVEL > 0: print("Total sequences: {}".format(len(data_list)))
    full_data = np.array(data_list)
    if DEBUG_LEVEL > 0: print("Full dataset size: {}".format(full_data.shape[0]))
    train_data, test_data = train_test_split(full_data, test_size=args.test_set_size, random_state=args.seed)
    return train_data, test_data

def main(args):
    # Part1: Load all the csv files from the zip file to a dataframe
    # Load the zip file from the given path
    zf = zipfile.ZipFile(args.data_path)
    # Get list of files contained within the archive
    files = zf.infolist()
    train_data, test_data = None, None
    validation_data = np.zeros(1)
    if args.split_mode == 'last':
        train_data, test_data = dataLastSplit(zf, files, args)
    elif args.split_mode == 'first' or args.split_mode == 'both':
        train_data, test_data = dataFirstSplit(zf, files, args)
    if args.split_mode == 'both':
        train_data, validation_data = train_test_split(train_data, test_size=test_data.shape[0], random_state=args.seed)
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
