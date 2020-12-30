# Deep Residual Mixture Model for Motion Capture

This codebase is used for training Deep Residual Mixture Models for mimicking motion capture data. The code is structured as follows:

* DRMM.py: Contains the base code for the DRMM. This was copied over unmodified from the original DRMM repository which can be found at https://github.com/PerttuHamalainen/DRMM
* Mocap_DRMM.py: Contains the main functionality for training the motion capture models.
* utils.py: Contains some functions related to the motion capture model that were moved over to reduce clutter in the main file, such as definition and creation of different model architectures.
* Mocap_data_preprocessor.py: Contains the code for preprocessing motion capture sequences into a dataset suitable for training the model. For more details, see the chapter "Data Preprocessing" below.
* motion_graph_data_generator.py: Can be used to generate training datasets from the motion capture csv-files using a motion graph based method similar to that used by Kovar et al. (https://graphics.cs.wisc.edu/Papers/2002/KGP02/mograph.pdf) Note: The current implementation is unoptimized and inefficient and thus not yet suitable for large scale dataset creation.
* visualization_utils.py: Contains the Skeleton class and various functions used to visualize the motion capture animations.
* visualize_csv.py: Contains code for visualizing the unprocessed motion data from the csv-files. Used for debugging purposes.

The file requirements.txt contains all the libraries that should be available in order to run the code.

## Data Preprocessing

The code in the file Mocap_data_preprocessor.py can be used in order to generate data for training the model. As a starting point, a zip-archive containing the motion capture sequences as csv-files is required. The code will by default look for it in the path "data/mocap-csv.zip", but this can be modified as a command line parameter.

Each csv-file included in the archive should contain just a single continuous motion capture sequence with each row describing a single frame, in order from top to bottom. The formatting of these files is rather particular: columns should be separated by a semicolon, with the first column containing information on whether the frame described by the row is the first one in the sequence (1) or not (0) and the last (i.e. 22nd) column being an unused column. Though, as these columns are currently discarded during processing, their actual contents are not important. The remaining 20 columns should contain the 3D positions (x, y, z) and rotations (in degrees around the respective axes) of various joints commonly used for humanoid character animation. (For the ordering of these columns refer to the global variable "CSV_COLUMNS" in the pre-processor code, where it can also be modified if necessary.) Positions and rotations should be provided as floating point values with commas separating each value from another.

By default, the pre-processor performs the following operations on the provided motion capture animations: First, each file is partitioned into non-overlapping sequences of 64 frames in length. If a file has fewer than 64 frames in total, it is discarded. Then, for each resulting sequence the following operations are performed:

1. Unused columns are dropped.
2. Columns are reordered so that connected joints are closer together.
3. Each column is split into six parts (three for position, three for rotation) and the rotations are dropped.
4. The sequence's position is normalized such that the x and z coordinates of the root joint (i.e. hips) are both zero for the first frame.
5. The sequence's rotation is normalized such that a vector connecting the left shoulder to the right shoulder is aligned with the positive z axis for the first frame.

During the process the dataset is also separated into two to three separate sets of sequences (train, test, and, optionally, validation).

The pre-processor supports the following command line arguments for customising the process:

* data-path: Relative path to the zip-archive containing the mocap animation csv-files. Default value is: data/mocap-csv.zip
* output-path: Relative path for the npz-file containing the resulting sequences as numpy arrays. Defaults to: data/mocap-dataset.npz
* seq-length: The sequence length, that is, how many frames should be in each resulting sequence. Defaults to: 64
* step-size: The number of frames that should separate the first frame of each sequence created from the same file. If the value is smaller than seq-length, overlapping sequences are produced. Defaults to: 64
* split-mode: Whether the data should be partitioned into test and train sets before or after the sequences are created. If this parameter is set to "first" the files are separated into two sets, one of which is turned into sequences for the train set, the other for the test set. If set to "last" the sequences are created first and are then separated into train and test sets. Note that using "last" will likely result in the same frames appearing in both the train and test sets, if step-size is smaller than seq-length. Additionally, this parameter supports the value "both". When it is used, the files are first separated into two batches for creating the train and test sets, and after the files have been processed into sequences, the sequences in the train set are further divided into train and validation sets in such a way that there are an equal number of sequences in both the test and validation sets. If the mode is not set to "both", a validation set will still be included in the resulting dataset, but it will only contain a single zero value. Defaults to: last
* test-size: What fraction of the data to set aside for the test set. Note that the behaviour of this value depends on the split mode: if split-mode is set to "first" or "both", this fraction is taken from the motion capture csv-files, whereas when split-mode is set to "last" it is taken from the final sequence count. Thus, if motion capture animations with variable lengths are used, the number of sequences in the test set may not as closely reflect the fraction over the total number of sequences when data is split first. Defaults to: 0.1
* mirror-animations: If this flag is set, the dataset is additionally augmented by including also the left-right mirroring of each sequence in the resulting dataset. By default, sequences are not mirrored.

## Training, Testing and Sampling the Model

When a dataset is available, the file Mocap_DRMM.py can be used to train, test and sample models.
The file supports the following command line parameters:

* data-path: Relative path to the file containing the dataset to be used for the indicated procedures. Defaults to: data/mocap-dataset.npz
* model-filename: Relative path and filename prefix for saving and loading model. Defaults to: models/mocap_model
* batch-size: The batch size used for training a model. Defaults to: 64
* iter-count: Over how many iterations should the model be trained. Defaults to: 20000
* seq-length: The number of frames in the input and output sequences for the network. Note that this value should be modified if and only if it was also modified for dataset creation. Defaults to: 64
* data-dim: The dimensionality of the input and output data, that is, the number of values used to describe each frame in a sequence. Note that this value should not be modified unless the code for the dataset creation has been modified. Defaults to: 60
* shuffle-buffer: The size of the buffer used for shuffling the dataset during training. Defaults to: 10000
* keyframe-count: How many "keyframes" should be used when taking conditioned samples from the model. Note that this is in addition to the first frame of a sequence, which is always designated as a keyframe. Defaults to: 2
* keyframe-mode: How to determine which frames to use as keyframes. Two modes are supported: "fixed" and "calculated". If set to "fixed", the keyframes will be placed throughout the animation at even intervals. If set to "calculated", the keyframes will be selected individually for each conditioning sequence based on a rate of change between frames calculated from the joint coordinates. Defaults to: fixed
* keyframe-calculation-interval: If using calculated keyframes, how many frames backward and forward to look for when calculating the rate of change. Defaults to: 2
* keyframe-display-mode: How should conditioning positions for keyframes be displayed in visualizations (if applicable). If set to "next", only the position for the next keyframe is displayed and it will disappear after that frame has been reached. If set to "all", positions for each keyframe are displayed at all times. Defaults to: next
* track-joints: The names of all joints to be used as conditioning throughout the sample. Multiple joint names should be separated by commas. If set to "none", no joints will be used for conditioning. Trajectories for tracked joints will be displayed in visualizations. Defaults to: none
* sample-mode: How to sample the model. Four options are supported: unconditioned, conditioned, extremes and none. If set to "unconditioned", model will be sampled without any conditioning. If set to "conditioned", a sequence from a set designated by the parameter "sample-set" will be used for conditioning the sample as defined by the joint tracking and keyframe related parameters. If set to "extremes", all sequences in the designated set will be used as conditioning in turn and results will be displayed for five sequences with the smallest and greatest error each. Note that the "extremes" option currently does not support all visualization enhancements, such as joint tracking. If set to "none", the model will not be sampled. Defaults to: unconditioned
* sample-set: Which set of data to use for conditioning samples (if applicable). Options are "train", "test", and "validation". Defaults to: test
* sample-batch-size: Batch size to use for sampling, that is, how many results to generate for each sample. This value is also used during testing the model. Defaults to: 32
* sample-out: Filename for saving the visualization animation from sampling the model. Note that animations are saved in the "animations" folder. Defaults to: animation.gif
* sample-cutoff: When generating samples, how many of the top results by likelihood should be considered. This values is used when generating conditioned samples or when testing the network. Note that out of the samples designated by this number, the one with the smallest error will be selected. Defaults to: 10
* shuffle-conditions: If this flag is set, the set used for conditioning will be shuffled before sampling. Useful when a random conditioned sample should be produced.
* temperature: The temperature to use when sampling the model. Defaults to: 1.0
* train-mode: Defines whether a new model should be trained. Supported values are "yes", "no", and "auto". The values "yes" and "no" can be used to force or prevent the training of a new model, respectively, whereas if set to "auto" a new model will only be trained if the model files designated by the "model-filename" parameter do not already exist. Defaults to: auto
* model-type: What model architecture to use. See the function "createModel" in utils.py for all available options. Note that the correct model should be designated even when loading a pre-existing model. Defaults to: baseline
* test-mode: Designates, which set of sequences the model should be tested on. Supported options are "test", "validation", and "none". Defaults to: none
* error-mode: Indicates whether error should be calculated based on all frames in the conditioning sequence ("all") or only the designated keyframes ("keypoints"). Defaults to: all
* error-calculation. Indicates whether errors for joint position should be calculated using L1 or L2 error. Defaults to: L2
* animation-type: Defines whether samples should be visualized as skeletons with joints connected by lines ("skeleton") or as a scatter plot with joints only shown as points ("scatter"). Defaults to: skeleton
* axis-type: How to set axis limits when visualizing the animation. If set to "full", the limits will be set such that the whole animation fits into the same view. If set to "centered", the visualization will be centered on the current frame and will follow the root joint. Defaults to: full
* axis-display: How much information to display for the axes. If set to "full", axis ticks, values and grid lines will be displayed, whereas if set to "minimal", they will be hidden. Defaults to: minimal
* no-plot: If this flag is set, the resulting sample will not be displayed on screen after creating the visualization is complete.
* seed: The random seed used for initializing random number generation during execution. Defaults to current system time, that is, the value returned by int(time.time())
* debug: Setting this flag enables some additional runtime prints for debugging purposes.
