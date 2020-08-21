''' Loads the CMU Mocap data from a zipfile, preprocesses it according to
    specifications and saves it as a numpy file.
'''
import os
import pandas as pd
import numpy as np
import zipfile

from sklearn.model_selection import train_test_split

# General parameters
DEBUG_LEVEL = 1

# File parameters
DATA_PATH = 'data/CMU-Mocap-csv.zip' # Path to the zip file containing the data
OUTPUT_PATH = 'data/cmu-numpy.npz' # Path to file where the data should be saved

# Data parameters
CSV_COLUMNS = ['is_first', 'hips', 'spine', 'left_upper_leg', 'left_lower_leg',
    'left_foot', 'right_upper_leg', 'right_lower_leg', 'right_foot', 'left_shoulder',
    'left_upper_arm', 'left_lower_arm', 'left_hand', 'left_toes', 'right_toes',
    'right_shoulder', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'head', 'neck', 'none']
ROOT_COLUMN = 'hips'         # column label for the body part to use as root
DROP_COLUMNS = ['is_first', 'none']
SHOULDER_COLUMNS = ['left_shoulder', 'right_shoulder']
CSV_DELIMITER = ';'
NORMALIZE_POSITION = True
NORMALIZE_ROTATION = True

# Preprocessor Parameters
SEQUENCE_LENGTH = 64    # The desired sequence length the data should be cut to
STEP_SIZE = 64          # Number of steps to offset sequences pulled from same animation
TEST_SET_SIZE = 0.1     # The relative size of the test set

# Takes a numpy array of shape [N, 3M] and rotates each [1,3] subset around the y-axis
def rotateMatrix(matrix, radians):
    if DEBUG_LEVEL > 1: print("Original matrix:\n{}".format(matrix))
    cosine = np.cos(radians)
    cos_matrix = np.resize(np.array([cosine, 1, cosine]), matrix.shape)
    if DEBUG_LEVEL > 1: print("Cosine matrix:\n{}".format(cos_matrix))
    rot_matrix = np.multiply(cos_matrix, matrix)
    if DEBUG_LEVEL > 1: print("Intermediate matrix:\n{}".format(rot_matrix))
    aux_matrix = np.zeros(matrix.shape)
    sine = np.sin(radians)
    aux_matrix[:, 0::3] = -sine * np.copy(matrix[:, 2::3])
    aux_matrix[:, 2::3] = sine * np.copy(matrix[:, 0::3])
    if DEBUG_LEVEL > 1: print("Auxiliary matrix:\n{}".format(aux_matrix))
    rot_matrix = np.add(rot_matrix, aux_matrix)
    if DEBUG_LEVEL > 1: print("Rotated matrix:\n{}".format(rot_matrix))
    return rot_matrix

def processDataFrame(dataFrame):
    # Drop unused columns
    for column in DROP_COLUMNS:
        if column in dataFrame:
            dataFrame = dataFrame.drop(column, axis=1)
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
    if NORMALIZE_ROTATION:
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
        # Calculate angle between vector and z axis (in radians)
        radians = np.arccos(shoulder_vector.dot(axis)/(np.linalg.norm(shoulder_vector)))
        if DEBUG_LEVEL > 1: print(radians)
        processedFrame = rotateMatrix(processedFrame, -radians)
        if DEBUG_LEVEL > 1: print(processedFrame[0])
    # Return the processed frame
    return processedFrame

def main():
    # Part1: Load all the csv files from the zip file to a dataframe
    # Load the zip file from the given path
    zf = zipfile.ZipFile(DATA_PATH)
    # Get list of files contained within the archive
    files = zf.infolist()
    # Initiate an empty list for storing the data
    data_list = []
    # Loop over all the files in the archive
    for f in files:
        # Make sure that the file is a CSV file
        if f.filename.endswith(".csv"):
            if (DEBUG_LEVEL > 0): print(f.filename)
            # Load the contents of the file into a dataframe
            dataFrame = pd.read_csv(zf.open(f.filename), sep=CSV_DELIMITER, names=CSV_COLUMNS)
            # Skip the file if the sequence is too short overall
            animationLength = dataFrame.count(axis=0)[0]
            if (animationLength < SEQUENCE_LENGTH):
                if (DEBUG_LEVEL > 0): print("Animation too short. Skipping...")
                continue
            # Split dataframe into multiple sequences and process individually
            start = 0
            end = SEQUENCE_LENGTH-1
            while end < animationLength:
                currentFrame = dataFrame.copy().truncate(before=start, after=end)
                # Reindex rows
                currentFrame.index = range(len(currentFrame.index))
                if (DEBUG_LEVEL > 1): print(currentFrame)
                dataArray = processDataFrame(currentFrame)
                #if (DEBUG_LEVEL > 1): print(dataArray)
                #dataArray = dataFrame.values
                if (DEBUG_LEVEL > 1): print(dataArray)
                # Store the dataframe in a list
                data_list.append(dataArray)
                start += STEP_SIZE
                end += STEP_SIZE
    if (DEBUG_LEVEL > 0): print(len(data_list))
    full_data = np.array(data_list)
    if DEBUG_LEVEL > 0: print("Full dataset size: {}".format(full_data.shape[0]))
    train_data, test_data = train_test_split(full_data, test_size=TEST_SET_SIZE)
    if DEBUG_LEVEL > 0: print("Train dataset size: {}".format(train_data.shape[0]))
    if DEBUG_LEVEL > 0: print("Test dataset size: {}".format(test_data.shape[0]))
    # Save the arrays into a npy file
    np.savez(OUTPUT_PATH, train_data=train_data, test_data=test_data)

if __name__ == '__main__':
    main()
