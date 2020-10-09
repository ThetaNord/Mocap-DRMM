import sys, time, argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

import DRMM
from DRMM import DRMMBlockHierarchy, dataStream, DataIn

# Visualization libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        dest='data_path',
        help='relative path to the npz file containing training and testing data',
        default='data/cmu-numpy.npz',
        type=str
    )
    parser.add_argument(
        '--model-filename',
        dest='model_filename',
        help='relative path and filename prefix for saving the final model',
        default='models/mocap_model',
        type=str
    )
    parser.add_argument(
        '--iter-count',
        dest='iteration_count',
        help='over how many iterations should the network be trained',
        default=20000,
        type=int
    )
    parser.add_argument(
        '--seq-length',
        dest='sequence_length',
        help='how many time steps per sequence',
        default=64,
        type=int
    )
    parser.add_argument(
        '--data-dim',
        dest='data_dimension',
        help='how many data points in sequence per time step',
        default=60,
        type=int
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        help='batch size to be used for training',
        default=64,
        type=int
    )
    parser.add_argument(
        '--shuffle-buffer',
        dest='shuffle_buffer',
        help='the size of the shuffle buffer for the test dataset',
        default=10000,
        type=int
    )
    parser.add_argument(
        '--keyframe-count',
        dest='keyframe_count',
        help='how many keyframes (besides first frame) should be used',
        default=2,
        type=int
    )
    parser.add_argument(
        '--keyframe-mode',
        dest='keyframe_mode',
        help='how to determine keyframes (fixed/calculated)',
        default='fixed',
        type=str
    )
    parser.add_argument(
        '--keyframe-display-mode',
        dest='keyframe_display_mode',
        help='how to show keyframes in animations (next/all)',
        default='next',
        type=str
    )
    parser.add_argument(
        '--keyframe-calculation-interval',
        dest='keyframe_calculation_interval',
        help='if calculating keframes, how many frames backwards and forwards to look',
        default=2,
        type=int
    )
    parser.add_argument(
        '--track-joint',
        dest='track_joint',
        help='name of the joint to track through a conditioned sample (defaults to none)',
        default='none',
        type=str
    )
    parser.add_argument(
        '--sample-mode',
        dest='sample_mode',
        help='how to sample the dataset (unconditioned/conditioned/extremes/none)',
        default='unconditioned',
        type=str
    )
    parser.add_argument(
        '--sample-set',
        dest='sample_set',
        help='which dataset to use in sampling (test/validation/train)',
        default='test',
        type=str
    )
    parser.add_argument(
        '--sample-batch-size',
        dest='sample_batch_size',
        help='batch size to be used when sampling the model',
        default=32,
        type=int
    )
    parser.add_argument(
        '--sample-cutoff',
        dest='sample_cutoff',
        help='how many samples from the batch to consider for evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--sample-out',
        dest='sample_outfile',
        help='filename for outputting the sample animation file',
        default='animation.gif',
        type=str
    )
    parser.add_argument(
        '--shuffle-conditions',
        dest='shuffle_conditions',
        help='if this flag is set, test set is shuffled before picking conditioning sample',
        action='store_true'
    )
    parser.add_argument(
        '--temperature',
        dest='temperature',
        help='the temperature to be used when sampling the model',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--train-mode',
        dest='train_mode',
        help='whether to train a new model or not (auto/yes/no)',
        default='auto',
        type=str
    )
    parser.add_argument(
        '--model-type',
        dest='model_type',
        help='what model architecture to use (baseline/extended)',
        default='baseline',
        type=str
    )
    parser.add_argument(
        '--test-mode',
        dest='test_mode',
        help='how to test the model (test/validation/none)',
        default='test',
        type=str
    )
    parser.add_argument(
        '--error-mode',
        dest='error_mode',
        help='for what points to calculate error in sampling and testing (all/keypoints)',
        default='all',
        type=str
    )
    parser.add_argument(
        '--error-calculation',
        dest='error_calculation',
        help='how to calculate error in sampling and testing (L1/L2)',
        default='L2',
        type=str
    )
    parser.add_argument(
        '--animation-type',
        dest='animation_type',
        help='how to animate samples (skeleton/scatter)',
        default='skeleton',
        type=str
    )
    parser.add_argument(
        '--axis-type',
        dest='axis_type',
        help='how to set axis limits on animation (centered/full)',
        default='centered',
        type=str
    )
    parser.add_argument(
        '--axis-display',
        dest='axis_display',
        help='how much information to display on axes (minimal/full)',
        default='minimal',
        type=str
    )
    parser.add_argument(
        '--no-plot',
        dest='no_plot',
        help='do not display plots for animation samples',
        action='store_true'
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        help='seed for initializing tensorflow random number generation',
        default=int(time.time()),
        type=int
    )
    parser.add_argument(
        '--debug',
        dest='debug',
        help='enable debug printing',
        action='store_true'
    )
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

#from .visualization import Skeleton
class Skeleton:

    joint_list = ['left_hand', 'right_hand', 'left_lower_arm',
        'right_lower_arm', 'left_upper_arm', 'right_upper_arm',
        'left_shoulder', 'right_shoulder', 'head', 'neck', 'spine', 'hips',
        'left_upper_leg', 'right_upper_leg', 'left_lower_leg', 'right_lower_leg',
        'left_foot', 'right_foot', 'left_toes', 'right_toes']
    connected_joints = [('hips', 'spine'), ('hips', 'left_upper_leg'),
        ('hips', 'right_upper_leg'), ('left_upper_leg', 'left_lower_leg'),
        ('left_lower_leg', 'left_foot'),  ('left_foot', 'left_toes'),
        ('right_upper_leg', 'right_lower_leg'), ('right_lower_leg', 'right_foot'),
        ('right_foot', 'right_toes'), ('spine', 'neck'), ('neck', 'head'),
        ('neck', 'left_upper_arm'), ('left_upper_arm', 'left_lower_arm'),
        ('left_lower_arm', 'left_hand'), ('neck', 'right_upper_arm'),
        ('right_upper_arm', 'right_lower_arm'), ('right_lower_arm', 'right_hand')]

    def __init__(self, joint_array, color='tab:blue', alpha=0.8):
        #self.connections = [(joint_list.index(connection[0]), joint_list.index(connection[1])) for connection in connected_joints]
        self.joint_sequence = joint_array
        self.root_node = 'hips'
        self.color = color
        self.alpha = alpha

    def get_joint_child(self, joint_name):
        return None

    def get_joint_position(self, joint_name, t, animation_index=0):
        joint_index = self.joint_list.index(joint_name)*3
        return self.joint_sequence[animation_index, t, joint_index:joint_index+3]

    def get_all_joint_positions(self, t, animation_index=0):
        xs = self.joint_sequence[animation_index, t, 0::3]
        ys = self.joint_sequence[animation_index, t, 1::3]
        zs = self.joint_sequence[animation_index, t, 2::3]
        return xs, ys, zs

    def animate_joints(self, t, graph, axis, animation_index=0):
        xs, ys, zs = self.get_all_joint_positions(t, animation_index)
        graph._offsets3d = (xs, zs, ys)
        if axis != None:
            origin = self.get_joint_position(self.root_node, t, animation_index)
            axis.set_xlim3d([origin[0]+1.0, origin[0]-1.0])
            axis.set_ylim3d([origin[2]+1.0, origin[2]-1.0])

    def animate_skeleton(self, t, lines, axis, animation_index=0):
        if axis != None:
            origin = self.get_joint_position(self.root_node, t, animation_index)
            axis.set_xlim3d([origin[0]+0.75, origin[0]-0.75])
            axis.set_ylim3d([origin[2]+0.75, origin[2]-0.75])
            axis.set_zlim3d([0.0, 1.5])
        for i, (start_joint, end_joint) in enumerate(self.connected_joints):
            start_position = self.get_joint_position(start_joint, t, animation_index)
            end_position = self.get_joint_position(end_joint, t, animation_index)
            positions = np.stack((start_position, end_position))
            xs, ys, zs = positions[:,0], positions[:,1], positions[:,2]
            lines[i].set_data(xs, zs)
            lines[i].set_3d_properties(ys)

def animateMultipleSkeletons(t, skeletons, lines_list, axes, animation_indices=None):
    if animation_indices is None:
        animation_indices = [0 for s in skeletons]
    for skeleton, lines, axis, index in zip(skeletons, lines_list, axes, animation_indices):
        skeleton.animate_skeleton(t, lines, axis, index)

def animateMultipleScatters(t, skeletons, graphs, axes, animation_indices=None):
    if animation_indices is None:
        animation_indices = [0 for s in skeletons]
    for skeleton, graph, axis, index in zip(skeletons, graphs, axes, animation_indices):
        skeleton.animate_joints(t, graph, axis, index)

def getAxisLimits(skeletons, animation_index=0):
    x0, x1, y0, y1, z0, z1 = None, None, None, None, None, None
    for skeleton in skeletons:
        xs = skeleton.joint_sequence[animation_index, :, 0::3]
        ys = skeleton.joint_sequence[animation_index, :, 1::3]
        zs = skeleton.joint_sequence[animation_index, :, 2::3]
        if x0 == None or x0 > np.min(xs): x0 = np.min(xs)
        if x1 == None or x1 < np.max(xs): x1 = np.max(xs)
        if y0 == None or y0 > np.min(ys): y0 = np.min(ys)
        if y1 == None or y1 < np.max(ys): y1 = np.max(ys)
        if z0 == None or z0 > np.min(zs): z0 = np.min(zs)
        if z1 == None or z1 < np.max(zs): z1 = np.max(zs)
    rnge = np.max([x1-x0, y1-y0, z1-z0])
    if x1-x0 < rnge:
        diff = rnge-(x1-x0)
        x0 -= diff/2
        x1 += diff/2
    if y1-y0 < rnge:
        diff = rnge-(y1-y0)
        y0 -= diff/2
        y1 += diff/2
    if z1-z0 < rnge:
        diff = rnge-(z1-z0)
        z0 -= diff/2
        z1 += diff/2
    return x0, x1, y0, y1, z0, z1

def cleanAxis(ax):
    ax.grid(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def loadDataset(args):
    # Load the data from the provided .npz file
    data_array = np.load(Path(args.data_path))
    # Convert into a Tensorflow dataset
    train_data = data_array['train_data']
    validation_data = data_array['validation_data']
    test_data = data_array['test_data']
    train_placeholder = tf.placeholder(train_data.dtype, train_data.shape)
    validation_placeholder = tf.placeholder(validation_data.dtype, validation_data.shape)
    test_placeholder = tf.placeholder(test_data.dtype, test_data.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_placeholder)
    validation_dataset = tf.data.Dataset.from_tensor_slices(validation_placeholder)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_placeholder)
    train_dict = {train_placeholder: train_data}
    validation_dict = {validation_placeholder: validation_data}
    test_dict = {test_placeholder: test_data}
    # Shuffle the training dataset
    train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed, reshuffle_each_iteration=True).repeat().batch(args.batch_size)
    return train_dataset, validation_dataset, test_dataset, train_dict, validation_dict, test_dict

def getTestItems(data_path, indices):
    # Load the data from the provided .npz file
    data_array = np.load(Path(data_path))
    items = data_array['test_data'][indices]
    print(items)
    return items

def calculateMinimumError(samples, targets, args, masks=None):
    if args.error_mode == "keypoints":
        if masks is None:
            print("ERROR: masks must be provided for keypoint error calculation")
            return
        samples = np.multiply(masks, samples)
        targets = np.multiply(masks, targets)
    errors = np.zeros(1)
    if args.error_calculation == 'L1':
        errors = np.sum(np.absolute(np.subtract(samples[:args.sample_cutoff], targets[:args.sample_cutoff])).reshape(args.sample_cutoff, args.sequence_length*args.data_dimension), axis=1)
    elif args.error_calculation == 'L2':
        errors = np.sum(np.square(np.subtract(samples[:args.sample_cutoff], targets[:args.sample_cutoff])).reshape(args.sample_cutoff, args.sequence_length*args.data_dimension), axis=1)
    min_error = np.min(errors)
    min_index = np.where(errors == min_error)[0][0]
    return min_error, min_index, errors

def getKeyFrameTimesteps(animation, args):
    timesteps = None
    if args.keyframe_mode == 'fixed':
        timesteps = getFixedKeyframes(args.sequence_length, args.keyframe_count)
    elif args.keyframe_mode == 'calculated':
        timesteps = calculateKeyFrames(animation, args.keyframe_count, args.keyframe_calculation_interval)
    if args.debug:
        print(timesteps)
    return timesteps

def getFixedKeyframes(sequence_length, keyframe_count):
    timesteps = [0]
    for i in range(1,keyframe_count):
        next_frame = i*sequence_length//keyframe_count
        timesteps.append(next_frame)
    timesteps.append(sequence_length-1)
    return timesteps

# Calculate the best keyframes for a sequence based on change rates
def calculateKeyFrames(sequence, keyframe_count, interval):
    change_rates = np.zeros(sequence.shape[0])
    # Calculate change rates for each frame except first and last
    for i in range(interval,sequence.shape[0]-interval):
        current_frame = sequence[i]
        prev_changes = np.subtract(current_frame, sequence[i-interval])
        next_changes = np.subtract(current_frame, sequence[i+interval])
        total_changes = np.absolute(prev_changes + next_changes)
        change_rates[i] = np.sum(total_changes)
    #print(change_rates)
    timesteps = np.zeros(1)
    idx = np.argpartition(change_rates, -keyframe_count-1)
    timesteps = np.sort(np.concatenate((timesteps, idx[-(keyframe_count-1):])))
    timesteps = np.append(timesteps, [sequence.shape[0]-1]).astype(int)
    return timesteps

def createModel(session, train, args):
    model = None
    if args.model_type == "baseline":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":7,"stride":2},   #input seq. length 32, output length 16
                {"nClasses":256,"nLayers":3,"kernelSize":7,"stride":2},   #in 16, out 8
            ],
            lastBlockClasses=256,
            lastBlockLayers=4,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "extended":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":6,"kernelSize":5,"stride":2},   #input seq. length 64, output length 32
                {"nClasses":256,"nLayers":8,"kernelSize":5,"stride":2},   #in 32, out 16
                {"nClasses":256,"nLayers":10,"kernelSize":5,"stride":2},   #in 16, out 8
            ],
            lastBlockClasses=256,
            lastBlockLayers=10,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.002)
    elif args.model_type == "extended-lite":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":3,"kernelSize":7,"stride":2},   #input seq. length 64, output length 32
                {"nClasses":256,"nLayers":4,"kernelSize":7,"stride":2},   #in 32, out 16
                {"nClasses":256,"nLayers":5,"kernelSize":7,"stride":2},   #in 16, out 8
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "large-kernel":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":3,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":9,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "large-kernel-extended":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":3,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":9,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "large-kernel-extra-layer":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":3,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":6,"kernelSize":9,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=6,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.004)
    elif args.model_type == "xl-kernel":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":3,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "deep-stack":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":3,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":6,"kernelSize":7,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=6,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "supersized":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":4,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":6,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":8,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":10,"kernelSize":11,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=10,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.004)
    elif args.model_type == "all-fives":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    return model

def testModel(model, test_dataset, test_dict, session, args):
    # Create iterator for the test dataset
    test_iterator = test_dataset.make_initializable_iterator()
    next_element = test_iterator.get_next()
    session.run(test_iterator.initializer, feed_dict=test_dict)
    # Iterate over the test set to calculate total error
    errors = []
    while True:
        try:
            # Get next element from test set, resize to sampling input
            target = session.run(next_element)
            samplingInputData = np.resize(target, [args.sample_batch_size, args.sequence_length, args.data_dimension])
            # Define timesteps which condition samples
            waypointTimesteps = getKeyFrameTimesteps(target, args)
            samplingMask = np.zeros_like(samplingInputData)
            samplingMask[:,waypointTimesteps,:] = 1.0
            if args.debug:
                print("First input sequence: {}".format(samplingInputData[0]))
                print("Last input sequence: {}".format(samplingInputData[-1]))
                print("Total difference: {}".format(np.sum(np.subtract(samplingInputData[0], samplingInputData[-1]))))
            # Sample the model
            samples = model.sample(inputs=DataIn(data=samplingInputData, mask=samplingMask),
                                    temperature=args.temperature, sorted=True)
            # Calculate the error
            min_error, _, _ = calculateMinimumError(samples, samplingInputData, args, samplingMask)
            if args.debug:
                print("Minimum error: {}".format(min_error))
            errors.append(min_error)
        except tf.errors.OutOfRangeError:
            break
    total_error = np.sum(errors)
    average_error = total_error/len(errors)
    print("Total error: {}\nAverage error: {}".format(total_error, average_error))

def sampleModel(model, args, condition_sample=None):
    # Sample from the model
    samples = None
    sample_errors = None
    best_index = 0
    waypoint_samples = []
    if args.sample_mode == "unconditioned":
        samples = model.sample(args.sample_batch_size, temperature=args.temperature, sorted=True)
        if args.debug: print(samples)
    elif args.sample_mode == "conditioned":
        if condition_sample is None:
            print("ERROR: Condition sample required!")
            return
        waypointTimesteps = getKeyFrameTimesteps(condition_sample, args)
        if args.keyframe_display_mode == 'next':
            waypoint_sample = np.zeros_like(condition_sample)
            for i in range(1,len(waypointTimesteps)):
                waypoint_sample[waypointTimesteps[i-1]:waypointTimesteps[i]] = condition_sample[waypointTimesteps[i]]
            waypoint_sample[waypointTimesteps[-2]:] = condition_sample[-1]
            waypoint_samples.append(waypoint_sample)
        elif args.keyframe_display_mode == 'all':
            for i in waypointTimesteps:
                waypoint_sample = np.zeros_like(condition_sample)
                waypoint_sample[:] = condition_sample[i]
                waypoint_samples.append(waypoint_sample)
        samplingInputData = np.resize(condition_sample, (args.sample_batch_size, args.sequence_length, args.data_dimension))
        samplingMask = np.zeros_like(samplingInputData)
        samplingMask[:,waypointTimesteps,:] = 1.0
        if args.track_joint != 'none':
            base_idx = Skeleton.joint_list.index(args.track_joint)*3
            samplingMask[:,:,base_idx:base_idx+3] = 1.0
            if args.debug:
                print(args.track_joint)
                print(base_idx)
                print(samplingMask)
        samples = model.sample(inputs=DataIn(data=samplingInputData, mask=samplingMask),
                                temperature=args.temperature, sorted=True)
        min_error, best_index, sample_errors = calculateMinimumError(samples, samplingInputData, args, samplingMask)
        if args.debug:
            print("Most likely sample errors:\n{}".format(sample_errors))
            print("Smallest error: {}".format(min_error))
            print("Best index: {}".format(best_index))
    # Create a skeleton with the given samples
    skeleton = Skeleton(samples)
    condition_skeleton = None
    waypoints_skeletons = []
    if args.sample_mode == "conditioned":
        condition_skeleton = Skeleton(np.array([condition_sample]))
        for waypoint_sample in waypoint_samples:
            waypoint_skeleton = Skeleton(np.array([waypoint_sample]), color='tab:red', alpha=0.4)
            waypoints_skeletons.append(waypoint_skeleton)
        print("Best sample error: {}".format(sample_errors[best_index]))
    # Visualize a sample
    fig = plt.figure()
    ax1, ax2 = None, None
    skeletons = []
    graphs = []
    line_list = []
    axes = []
    animation_indices = [0, best_index]
    if args.sample_mode == "unconditioned":
        skeletons = [skeleton]
        # Create a single subplot
        ax1 = fig.add_subplot(111, projection='3d')
        # Set axis properties
        ax1.set_title('Sample Animation')
        if args.axis_display == 'full':
            ax1.set_xlabel('X')
            ax1.set_ylabel('Z')
            ax1.set_zlabel('Y')
        elif args.axis_display == 'minimal':
            cleanAxis(ax1)
        if args.axis_type == 'centered':
            ax1.set_xlim3d([1.0, -1.0])
            ax1.set_ylim3d([1.0, -1.0])
            ax1.set_zlim3d([0.0, 2.0])
            axes = [ax1]
        elif args.axis_type == 'full':
            x0, x1, y0, y1, z0, z1 = getAxisLimits(skeletons)
            ax1.set_xlim3d([x1+0.1, x0-0.1])
            ax1.set_ylim3d([z1+0.1, z0-0.1])
            ax1.set_zlim3d([0, (y1-y0)+0.2])
            axes = [None]
        if args.animation_type == 'scatter':
            # Get initial joint positions
            xs, ys, zs = skeleton.get_all_joint_positions(0)
            # Plot joints as a scatter plot
            graph = ax1.scatter(xs, zs, ys)
            graphs.append(graph)
        elif args.animation_type == 'skeleton':
            lines = []
            for x in skeleton.joint_list:
                line, = ax1.plot([],[],[], color=skeleton.color, alpha=skeleton.alpha)
                lines.append(line)
            line_list.append(lines)
    elif args.sample_mode == "conditioned":
        skeletons = [condition_skeleton, skeleton]
        # Make the plot wider
        fig.set_figwidth(12)
        # Create two subplots
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        # Set axis properties
        ax1.set_title('Original Animation')
        ax2.set_title('Conditioned Sample')
        if args.axis_display == 'full':
            ax1.set_xlabel('X')
            ax1.set_ylabel('Z')
            ax1.set_zlabel('Y')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Z')
            ax2.set_zlabel('Y')
        elif args.axis_display == 'minimal':
            cleanAxis(ax1)
            cleanAxis(ax2)
        if args.axis_type == 'centered':
            ax1.set_xlim3d([1.0, -1.0])
            ax1.set_ylim3d([1.0, -1.0])
            ax1.set_zlim3d([0.0, 2.0])
            ax2.set_xlim3d([1.0, -1.0])
            ax2.set_ylim3d([1.0, -1.0])
            ax2.set_zlim3d([0.0, 2.0])
            axes = [ax1, ax2]
        elif args.axis_type == 'full':
            axes = [None, None]
        if args.animation_type == 'scatter':
            # Get initial joint positions
            xs, ys, zs = condition_skeleton.get_all_joint_positions(0)
            graph1 = ax1.scatter(xs, zs, ys)
            # Get initial joint positions
            xs, ys, zs = skeleton.get_all_joint_positions(0)
            graph2 = ax2.scatter(xs, zs, ys)
            graphs = [graph1, graph2]
            for waypoint_skeleton in waypoints_skeletons:
                # Get initial joint positions
                xs, ys, zs = waypoint_skeleton.get_all_joint_positions(0)
                graph = ax2.scatter(xs, zs, ys, c='red', alpha=0.2)
                skeletons.append(waypoint_skeleton)
                graphs.append(graph)
                axes.append(None)
                animation_indices.append(0)
        elif args.animation_type == 'skeleton':
            temp_axes = [ax1, ax2]
            for waypoint_skeleton in waypoints_skeletons:
                skeletons.append(waypoint_skeleton)
                temp_axes.append(ax2)
                axes.append(None)
                animation_indices.append(0)
            for s, ax in zip(skeletons, temp_axes):
                lines = []
                for x in s.joint_list:
                    line, = ax.plot([],[],[], color=s.color, alpha=s.alpha)
                    lines.append(line)
                line_list.append(lines)
        if args.axis_type == 'full':
            x0, x1, y0, y1, z0, z1 = getAxisLimits(skeletons)
            ax1.set_xlim3d([x1+0.1, x0-0.1])
            ax1.set_ylim3d([z1+0.1, z0-0.1])
            ax1.set_zlim3d([0.0, (y1-y0)+0.2])
            ax2.set_xlim3d([x1+0.1, x0-0.1])
            ax2.set_ylim3d([z1+0.1, z0-0.1])
            ax2.set_zlim3d([0.0, (y1-y0)+0.2])
        if args.track_joint != 'none':
            base_idx = Skeleton.joint_list.index(args.track_joint)*3
            joint_path = condition_skeleton.joint_sequence[0, :, base_idx:base_idx+3]
            xs = np.repeat(joint_path[:, 0::3], 2)[1:-1]
            ys = np.repeat(joint_path[:, 1::3], 2)[1:-1]
            zs = np.repeat(joint_path[:, 2::3], 2)[1:-1]
            ax2.plot(xs, zs, ys, color='tab:red', alpha=0.4)
    # Create the Animation object
    skeleton_animation = None
    if args.animation_type == 'skeleton':
        skeleton_animation = animation.FuncAnimation(fig, animateMultipleSkeletons,
                                        64, fargs=(skeletons, line_list, axes, animation_indices), interval=33, blit=False)
    elif args.animation_type == 'scatter':
        skeleton_animation = animation.FuncAnimation(fig, animateMultipleScatters,
                                        64, fargs=(skeletons, line_list, axes, animation_indices), interval=33, blit=False)
    skeleton_animation.save('animations/'+args.sample_outfile, writer='imagemagick', fps=30)
    # Show plot
    if not args.no_plot:
        plt.show()

def showBestAndWorst(model, test_dataset, test_dict, session, args):
    # Create iterator for the test dataset
    test_iterator = test_dataset.make_initializable_iterator()
    next_element = test_iterator.get_next()
    session.run(test_iterator.initializer, feed_dict=test_dict)
    # Iterate over the test set to calculate errors
    errors = []
    best_samples = []
    targets = []
    waypoints = []
    while True:
        try:
            # Get next element from test set, resize to sampling input
            target = session.run(next_element)
            targets.append(target)
            samplingInputData = np.resize(target, [args.sample_batch_size, args.sequence_length, args.data_dimension])
            # Define timesteps which condition samples
            waypointTimesteps = getKeyFrameTimesteps(target, args)
            waypoints.append(waypointTimesteps)
            samplingMask = np.zeros_like(samplingInputData)
            samplingMask[:,waypointTimesteps,:] = 1.0
            if args.debug:
                print("First input sequence: {}".format(samplingInputData[0]))
                print("Last input sequence: {}".format(samplingInputData[-1]))
                print("Total difference: {}".format(np.sum(np.subtract(samplingInputData[0], samplingInputData[-1]))))
            # Sample the model
            samples = model.sample(inputs=DataIn(data=samplingInputData, mask=samplingMask),
                                    temperature=args.temperature, sorted=True)
            # Calculate the error
            min_error, min_index, _ = calculateMinimumError(samples, samplingInputData, args, samplingMask)
            if args.debug:
                print("Minimum error: {}".format(min_error))
                print("Minimum error index: {}".format(min_index))
            errors.append(min_error)
            best_samples.append(samples[min_index])
        except tf.errors.OutOfRangeError:
            break
    # Find best and worst samples
    errors = np.array(errors)
    samples = np.array(best_samples)
    targets = np.array(targets)
    waypoints = np.array(waypoints)
    best_example_indices = np.argpartition(errors, 170)[:5]
    worst_example_indices = np.argpartition(errors, -170)[-5:]
    bw_indices = np.concatenate((best_example_indices, worst_example_indices))
    print("Best errors:\n{}".format(errors[best_example_indices]))
    print("Worst errors:\n{}".format(errors[worst_example_indices]))
    if args.debug:
        print("All errors:\n{}".format(errors))
        print("Indice array:\n{}".format(bw_indices))
        print("Samples:\n{}".format(samples))
        print("Targets:\n{}".format(targets))
    # Create skeletons for the sample sequences
    sample_skeletons = [Skeleton(np.array([sample])) for sample in samples[bw_indices]]
    # Create skeletons for target sequences
    target_skeletons = [Skeleton(np.array([target])) for target in targets[bw_indices]]
    # Create skeletons for keypoint sequences
    waypoints_skeletons = []
    for target, waypointTimesteps in zip(targets[bw_indices], waypoints[bw_indices]):
        if args.debug:
            print("Target array: {}".format(target))
            print("Target array shape: {}".format(target.shape))
        waypoint_sample = np.zeros_like(target)
        for i in range(1,len(waypointTimesteps)):
            waypoint_sample[waypointTimesteps[i-1]:waypointTimesteps[i]] = target[waypointTimesteps[i]]
        waypoint_sample[waypointTimesteps[-2]:] = target[-1]
        waypoints_skeletons.append(Skeleton(np.array([waypoint_sample]), color='tab:red', alpha=0.4))
    # Visualize
    fig = plt.figure(figsize=(28, 10), dpi=100)
    skeletons = []
    graphs = []
    line_list = []
    axes = []
    animation_indices = [0 for x in range(10)]
    for idx, sample_skeleton, target_skeleton, waypoint_skeleton in zip(range(10), sample_skeletons, target_skeletons, waypoints_skeletons):
        # Create subplots
        ax1 = fig.add_subplot(2, 10, idx*2+1, projection='3d')
        ax2 = fig.add_subplot(2, 10, idx*2+2, projection='3d')
        # Set axis properties
        ax1.set_xlim3d([1.0, -1.0])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([1.0, -1.0])
        ax1.set_ylabel('Z')
        ax1.set_zlim3d([0.0, 2.0])
        ax1.set_zlabel('Y')
        ax1.set_title('Original')
        # Set axis properties
        ax2.set_xlim3d([1.0, -1.0])
        ax2.set_xlabel('X')
        ax2.set_ylim3d([1.0, -1.0])
        ax2.set_ylabel('Z')
        ax2.set_zlim3d([0.0, 2.0])
        ax2.set_zlabel('Y')
        ax2.set_title('Sample')
        if args.animation_type == 'scatter':
            # Plot initial joint positions
            xs, ys, zs = target_skeleton.get_all_joint_positions(0)
            graphs.append(ax1.scatter(xs, zs, ys))
            skeletons.append(target_skeleton)
            axes.append(ax1)
            xs, ys, zs = sample_skeleton.get_all_joint_positions(0)
            graphs.append(ax2.scatter(xs, zs, ys))
            skeletons.append(sample_skeleton)
            axes.append(ax2)
            xs, ys, zs = waypoint_skeleton.get_all_joint_positions(0)
            graphs.append(ax2.scatter(xs, zs, ys, c='red', alpha=0.2))
            skeletons.append(waypoint_skeleton)
            axes.append(None)
        elif args.animation_type == 'skeleton':
            for s, ax in zip([target_skeleton, sample_skeleton, waypoint_skeleton], [ax1, ax2, ax2]):
                lines = []
                for x in s.joint_list:
                    line, = ax.plot([],[],[], color=s.color, alpha=s.alpha)
                    lines.append(line)
                line_list.append(lines)
                skeletons.append(s)
            axes.append(ax1)
            axes.append(ax2)
            axes.append(None)
    # Create the Animation object
    skeleton_animation = None
    if args.animation_type == 'scatter':
        skeleton_animation = animation.FuncAnimation(fig, animateMultipleScatters,
                                        64, fargs=(skeletons, graphs, axes), interval=33, blit=False)
    elif args.animation_type == 'skeleton':
        skeleton_animation = animation.FuncAnimation(fig, animateMultipleSkeletons,
                                            64, fargs=(skeletons, line_list, axes), interval=33, blit=False)
    # Save animation
    skeleton_animation.save('animations/'+args.sample_outfile, writer='imagemagick', fps=30)
    # Show plot
    if not args.no_plot:
        plt.show()

def main(args):
    #Init tf
    tf.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(args.seed)
    # Load dataset
    train_dataset, validation_dataset, test_dataset, train_dict, validation_dict, test_dict = loadDataset(args)
    # Check whether a new model should be trained
    train = True if args.train_mode == "yes" else False
    # If train_mode is auto, check whether a model exists
    if args.train_mode == "auto":
        train = not Path(args.model_filename+".index").is_file()
    # Create model
    model = createModel(sess, train, args)
    print("Total model parameters: ", model.nParameters)
    # Train or load model
    saver = tf.train.Saver()
    if not train:
        saver.restore(sess, args.model_filename)
    else:
        # Initialize Tensorflow
        tf.global_variables_initializer().run(session=sess)
        # Initialize dataset iterator
        iterator = train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer, feed_dict=train_dict)
        # Data-driven init with a random batch
        model.init(sess.run(next_element))
        # Optimize
        for i in range(args.iteration_count):
            info = model.train(i/args.iteration_count, sess.run(next_element))
            if i % 100 == 0:
                print("Stage {}/{}, Iteration {}/{}, Loss {:.3f}, learning rate {:.6f}, precision {:.3f}".format(
                    info["stage"],info["nStages"],
                    i, args.iteration_count,
                    info["loss"],
                    info["lr"],
                    info["rho"]),end="\r")
        # Generate the directories for saving models if they do not already exist
        model_path = Path(args.model_filename).parent
        if not model_path.is_dir():
            model_path.mkdir(parents=True, exist_ok=True)
        saver.save(sess, args.model_filename)
    # Test model
    if args.test_mode != 'none':
        if args.test_mode == 'test':
            testModel(model, test_dataset, test_dict, sess, args)
        if args.test_mode == 'validation':
            testModel(model, validation_dataset, validation_dict, sess, args)
    # Sample from the model
    if args.sample_mode != "none":
        sample_dataset, sample_dict = None, None
        if args.sample_set == 'train': sample_dataset, sample_dict = train_dataset, train_dict
        elif args.sample_set == 'validation': sample_dataset, sample_dict = validation_dataset, validation_dict
        elif args.sample_set == 'test': sample_dataset, sample_dict = test_dataset, test_dict
        if args.sample_mode == "extremes":
            showBestAndWorst(model, sample_dataset, sample_dict, sess, args)
        else:
            if args.shuffle_conditions:
                sample_dataset = sample_dataset.shuffle(buffer_size=1000, seed=args.seed)
            sample_iterator = sample_dataset.make_initializable_iterator()
            next_element = sample_iterator.get_next()
            sess.run(sample_iterator.initializer, feed_dict=sample_dict)
            sampleModel(model, args, sess.run(next_element))

if __name__ == '__main__':
    # Parse command line arguments
    argv = sys.argv
    args = parse_args(argv)
    if args.debug:
        print(args)
    main(args)
