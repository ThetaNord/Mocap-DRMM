import sys, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import DRMM
from DRMM import DRMMBlockHierarchy, dataStream, DataIn

# Visualization libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

ITERATOR = None

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
        '--sample-batch-size',
        dest='sample_batch_size',
        help='batch size to be used when sampling the model',
        default=32,
        type=int
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
        help='whether to train a new model or not (yes/no/auto)',
        default='auto',
        type=str
    )
    parser.add_argument(
        '--no-test',
        dest='no_test',
        help='skip testing the network against the test set',
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

    def __init__(self, joint_array):
        print(joint_array.shape)
        self.joint_list = ['hips', 'spine', 'left_upper_leg', 'left_lower_leg',
            'left_foot', 'right_upper_leg', 'right_lower_leg', 'right_foot',
            'left_shoulder', 'left_upper_arm', 'left_lower_arm', 'left_hand',
            'left_toes', 'right_toes', 'right_shoulder', 'right_upper_arm',
            'right_lower_arm', 'right_hand', 'head', 'neck']
        self.joint_sequence = joint_array

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

    def animate_skeleton(self, t, graph, animation_index=0):
        xs, ys, zs = self.get_all_joint_positions(t, animation_index)
        graph._offsets3d = (xs, zs, ys)

def loadDataset(data_path):
    # Load the data from the provided .npz file
    data_array = np.load(Path(data_path))
    # Convert into a Tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(data_array['train_data'])
    test_dataset = tf.data.Dataset.from_tensor_slices(data_array['test_data'])
    # Shuffle the dataset
    train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True).repeat()
    return train_dataset, test_dataset

# Function for retrieving a single batch of training data
def getDataBatch(batch_size, next_element, tf_session):
    global ITERATOR
    dataBatch = []
    for i in range(batch_size):
        try:
            data = tf_session.run(next_element)
        except tf.errors.OutOfRangeError:
            tf_session.run(ITERATOR.initializer)
            data = tf_session.run(next_element)
            print(data)
        dataBatch.append(data)
    dataBatch = np.asarray(dataBatch)
    return dataBatch

def main(args):
    global ITERATOR
    #Init tf
    tf.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(args.seed)
    # Load dataset
    train_dataset, test_dataset = loadDataset(args.data_path)
    # Check whether a new model should be trained
    train = True if args.train_mode == "yes" else False
    # If train_mode is auto, check whether a model exists
    if args.train_mode == "auto":
        train = not Path(args.model_filename+".index").is_file()
    # Create model
    model = DRMMBlockHierarchy(sess,
        inputs=dataStream(dataType="continuous",shape=[None,args.sequence_length,args.data_dimension],useGaussianPrior=True,useBoxConstraints=True),
        blockDefs=[
            {"nClasses":256,"nLayers":2,"kernelSize":7,"stride":2},   #input seq. length 32, output length 16
            {"nClasses":256,"nLayers":3,"kernelSize":7,"stride":2},   #in 16, out 8
        ],
        lastBlockClasses=256,
        lastBlockLayers=4,
        train=train,    #if False, optimization ops will not be created, which saves some time
        initialLearningRate=0.005)
    print("Total model parameters: ", model.nParameters)

    # Train or load model
    saver = tf.train.Saver()
    if not train:
        saver.restore(sess, args.model_filename)
    else:
        # Initialize Tensorflow
        tf.global_variables_initializer().run(session=sess)
        # Initialize dataset iterator
        ITERATOR = train_dataset.make_initializable_iterator()
        next_element = ITERATOR.get_next()
        sess.run(ITERATOR.initializer)
        # Data-driven init with a random batch
        model.init(getDataBatch(args.batch_size, next_element, sess))
        # Optimize
        for i in range(args.iteration_count):
            info = model.train(i/args.iteration_count,
                            getDataBatch(args.batch_size, next_element, sess))
            if i % 10 == 0:
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
    if not args.no_test:
        waypointTimesteps = [0,args.sequence_length//2,args.sequence_length-1]
        samplingInputData = np.zeros([args.sample_batch_size, args.sequence_length, args.data_dimension])
        samplingMask = np.zeros_like(samplingInputData)
        samplingMask[:,waypointTimesteps,0] = 1.0
        test_iterator = test_dataset.make_one_shot_iterator()
        next_element = test_iterator.get_next()
        # Iterate over the test set to calculate total error
        errors = []
        while True:
            try:
                # Get next element from test set, resize to sampling input
                target = sess.run(next_element)
                samplingInputData = np.resize(target, samplingInputData.shape)
                # Sample the model
                samples = model.sample(inputs=DataIn(data=samplingInputData,mask=samplingMask),
                                        temperature=args.temperature, sorted=True)
                #print("Samples shape: {}".format(samples.shape))
                # Calculate the errors
                sample_errors = np.sum(np.square(samples - samplingInputData).reshape(args.sample_batch_size, args.sequence_length*args.data_dimension), axis=1)
                #print("Errors shape: {}".format(sample_errors.shape))
                min_distance = np.min(sample_errors[:10])
                #print("First 10 errors: {}".format(sample_errors[:10]))
                #print("Minimum distance: {}".format(min_distance))
                errors.append(min_distance)
            except tf.errors.OutOfRangeError:
                break
        total_error = np.sum(errors)
        average_error = total_error/len(errors)
        print("Total error: {}\nAverage error: {}".format(total_error, average_error))
    # Sample from the model
    samples = model.sample(args.sample_batch_size, temperature=args.temperature, sorted=True)
    if args.debug: print(samples)
    skeleton = Skeleton(samples)
    # Visualize a sample
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Set axis properties
    ax.set_xlim3d([1.0, -1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([1.0, -1.0])
    ax.set_ylabel('Z')
    ax.set_zlim3d([0.0, 2.0])
    ax.set_zlabel('Y')
    ax.set_title('Sample Animation')
    # Get initial joint positions
    xs, ys, zs = skeleton.get_all_joint_positions(0)
    # For now, just plot as points
    # TODO: Plot the actual skeleton
    #graph, = ax.plot(xs, ys, zs, linestyle="", marker="o")
    graph = ax.scatter(xs, zs, ys)
    # Create the Animation object
    skeleton_animation = animation.FuncAnimation(fig, skeleton.animate_skeleton,
                                        64, fargs=(graph,), interval=33, blit=False)
    skeleton_animation.save('animations/animation.gif', writer='imagemagick', fps=30)
    # Show plot
    plt.show()

if __name__ == '__main__':
    # Parse command line arguments
    argv = sys.argv
    args = parse_args(argv)
    if args.debug:
        print(args)
    main(args)
