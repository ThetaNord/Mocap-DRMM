import numpy as np

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
