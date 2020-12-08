import sys, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# Local imports
import visualization_utils
from visualization_utils import Skeleton, animateMultipleScatters, animateMultipleSkeletons, cleanAxis, getAxisLimits

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-path',
        dest='csv_path',
        help='relative path to the csv file to visualize',
        default='data/test.csv',
        type=str
    )
    parser.add_argument(
        '--outfile',
        dest='outfile',
        help='filename for outputting the animation as gif',
        default='csv_test.gif',
        type=str
    )
    parser.add_argument(
        '--animation-type',
        dest='animation_type',
        help='how to animate samples (skeleton/scatter)',
        default='scatter',
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
        default='full',
        type=str
    )
    parser.add_argument(
        '--mirror',
        dest='mirror',
        help='mirror animation',
        action='store_true'
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
        help='seed for initializing random number generation',
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

def main(args):
    # Load Dataframe
    data_frame = pd.read_csv(args.csv_path, sep=CSV_DELIMITER, names=CSV_COLUMNS)
    # Drop unused columns
    for column in DROP_COLUMNS:
        if column in data_frame:
            data_frame = data_frame.drop(column, axis=1)
    # Split all columns to six parts
    data_frame = data_frame.stack().str.extractall('([\d\.E-]+)').unstack([-2, -1])
    # Drop columns that contain rotations
    data_frame.columns = np.arange(len(data_frame.columns))
    drop_indices = np.arange(data_frame.count(axis=1)[0])
    drop_indices = np.where(drop_indices % 6 > 2)
    drop_indices = drop_indices[0].tolist()
    data_frame = data_frame.drop(drop_indices, axis='columns')
    # Convert all columnts to float
    data_frame = data_frame.astype(np.float64)
    # Convert from data frame to numpy array
    data_array = data_frame.values
    # Visualize data array as a scatter skeleton
    skeleton = Skeleton(np.array([data_array]))
    skeleton.root_node = 'left_hand'
    fig = plt.figure()
    skeletons = [skeleton]
    graphs = []
    line_list = []
    axes = []
    animation_indices = [0]
    animation_length = data_array.shape[0]
    print(animation_length)
    # Create a single subplot
    ax1 = fig.add_subplot(111, projection='3d')
    # Set axis properties
    ax1.set_title('CSV Animation')
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
    # Create the Animation object
    skeleton_animation = None
    if args.animation_type == 'skeleton':
        skeleton_animation = animation.FuncAnimation(fig, animateMultipleSkeletons,
                                        animation_length, fargs=(skeletons, line_list, axes, animation_indices), interval=33, blit=False)
    elif args.animation_type == 'scatter':
        skeleton_animation = animation.FuncAnimation(fig, animateMultipleScatters,
                                        animation_length, fargs=(skeletons, graphs, axes, animation_indices), interval=33, blit=False)
    #skeleton_animation.save('animations/'+args.outfile, writer='imagemagick', fps=30)
    # Show plot
    if not args.no_plot:
        plt.show()

if __name__ == '__main__':
    # Parse command line arguments
    argv = sys.argv
    args = parse_args(argv)
    if args.debug:
        print(args)
    main(args)
