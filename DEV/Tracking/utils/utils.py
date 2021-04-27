import os, copy, glob, glob2, numpy as np, colorsys
from numba import jit
from pyquaternion import Quaternion

def quaternion_yaw(q: Quaternion):
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def log_sum(weight_array):
    '''
    weight_sum = log_sum(weight_array)

    Sum of logarithmic components
    w_sum = w_smallest + log( 1 + sum(exp(w_rest - w_smallest)) )
    '''
    weight_array.sort()
    _w0 = weight_array[0]
    _wr = weight_array[1:]
    _wdelta = _wr - _w0
    _exp = np.exp(_wdelta)
    _sum = np.sum(_exp)
    _weight = _w0 + np.log(1 + _sum)
    
    return _weight


def normailized_heading(x): 
    """
    Input a heading, output a normalized heading
    heading [-pi, pi]
    """
    #x = x % (2 * np.pi)
    if x > np.pi: 
        return x - 2 * np.pi
    elif x < -np.pi:
        return x + 2 * np.pi
    return x

def angle_difference(x, y):
    """
    Get the smallest difference between 2 angles - x, y.
    Both the input and output are in radian.
    Output is in (-pi, pi] range.
    >>> angle_difference(math.pi, math.pi)
    0.0
    >>> angle_difference(math.pi, -math.pi)
    0.0
    >>> angle_difference(math.pi/2, -math.pi/2)
    3.141592653589793
    >>> angle_difference(math.pi/4, -math.pi/4)
    1.5707963267948966
    >>> angle_difference(math.pi/6, 11*math.pi/6)
    1.0471975511965983
    """
    diff = (x - y) % (2 * np.pi)
    if diff > np.pi:
        return diff - (2 * np.pi)
    else:
        return diff

def isstring(string_test):
    try:
        return isinstance(string_test, basestring)
    except NameError:
        return isinstance(string_test, str)

def is_path_creatable(pathname):
    '''
    if any previous level of parent folder exists, returns true
    '''
    if not is_path_valid(pathname): return False
    pathname = os.path.normpath(pathname)
    pathname = os.path.dirname(os.path.abspath(pathname))

	# recursively to find the previous level of parent folder existing
    while not is_path_exists(pathname):
    	pathname_new = os.path.dirname(os.path.abspath(pathname))
    	if pathname_new == pathname: return False
    	pathname = pathname_new
    return os.access(pathname, os.W_OK)
def is_path_valid(pathname):
    try:
        if not isstring(pathname) or not pathname: return False
    except TypeError: return False
    else: return True

def is_path_exists(pathname):
    try: return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError: return False

def is_path_exists_or_creatable(pathname):
    try: return is_path_exists(pathname) or is_path_creatable(pathname)
    except OSError: return False

def isfolder(pathname):
    '''
    if '.' exists in the subfolder, the function still justifies it as a folder. e.g., /mnt/dome/adhoc_0.5x/abc is a folder
    if '.' exists after all slashes, the function will not justify is as a folder. e.g., /mnt/dome/adhoc_0.5x is NOT a folder
    '''
    if is_path_valid(pathname):
    	pathname = os.path.normpath(pathname)
    	if pathname == './': return True
    	name = os.path.splitext(os.path.basename(pathname))[0]
    	ext = os.path.splitext(pathname)[1]
    	return len(name) > 0 and len(ext) == 0
    else: return False

def fileparts(input_path, warning=True, debug=True):
    '''
    this function return a tuple, which contains (directory, filename, extension)
    if the file has multiple extension, only last one will be displayed
    parameters:
    	input_path:     a string path
    outputs:
    	directory:      the parent directory
    	filename:       the file name without extension
    	ext:            the extension
    '''
    good_path = safe_path(input_path, debug=debug)
    if len(good_path) == 0: return ('', '', '')
    if good_path[-1] == '/':
    	if len(good_path) > 1: return (good_path[:-1], '', '')	# ignore the final '/'
    	else: return (good_path, '', '')	                          # ignore the final '/'    
    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    ext = os.path.splitext(good_path)[1]
    return (directory, filename, ext)


def safe_path(input_path, 
              warning=True, 
              debug=True):
    '''
    convert path to a valid OS format, e.g., 
        empty string '' to '.', remove redundant '/' at the 
        end from 'aa/' to 'aa'

    parameters:
    	input_path:		a string

    outputs:
    	safe_data:		a valid path in OS format
    '''
    if debug: assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data

def mkdir_if_missing(input_path, 
                     warning=True, 
                     debug=True):
	'''
	create a directory if not existing:
		1. if the input is a path of file, 
           then create the parent directory of this file
		2. if the root directory does not exists for the input, 
           then create all the root directories recursively 
           until the parent directory of input exists

	parameters:
		input_path:     a string path
	'''
	good_path = safe_path(input_path, warning=warning, debug=debug)
	if debug: assert is_path_exists_or_creatable(good_path), 'input path is not valid or creatable: %s' % good_path
	dirname, _, _ = fileparts(good_path)
	if not is_path_exists(dirname): mkdir_if_missing(dirname)
	if isfolder(good_path) and not is_path_exists(good_path): os.mkdir(good_path)


def format_sample_result(sample_token, tracking_name, tracker):
  '''
  Input:
    tracker: (9): [h, w, l, x, y, z, rot_y], tracking_id, tracking_score
  Output:
  sample_result {
    "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
    "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
    "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
    "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
    "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                       Note that the tracking_name cannot change throughout a track.
    "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                       We average over frame level scores to compute the track level score.
                                       The score is used to determine positive and negative tracks via thresholding.
  }
  '''
  rotation = Quaternion(axis=[0, 0, 1], angle=tracker[6]).elements
  sample_result = {
    'sample_token': sample_token,
    'translation': [tracker[3], tracker[4], tracker[5]],
    'size': [tracker[1], tracker[2], tracker[0]],
    'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
    'velocity': [0, 0],
    'tracking_id': str(int(tracker[7])),
    'tracking_name': tracking_name,
    'tracking_score': tracker[8]
  }

  return sample_result

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def convert_3dbox_to_8corner(bbox3d_input, nuscenes_to_kitti=False):
    """
    Takes an object and a projection matrix (P) and projects the 3d bounding box
    into the image plane. 
    Returns: 
        corners_2d: (8,2) array in the left image corrd, 
        corners_3d: (8,3) array in the rect camera corrd. 
    Note: 
        the output of this function will be passed to the function iou3d for 
        calculating the 3d-iou. But the function iou3d was written for the kitti
        so the caller needs to set nuscense_to_kitti to True is the input 
        bbox3d_input is in nuscense format. 
    """
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)
    if nuscenes_to_kitti:
          # transform to kitti format first
      bbox3d_nuscenes = copy.copy(bbox3d)
      # kitti:    [x,  y,  z,  a, l, w, h]
      # nuscenes: [y, -z, -x, -a, w, l, h]
      bbox3d[0] =  bbox3d_nuscenes[1]
      bbox3d[1] = -bbox3d_nuscenes[2]
      bbox3d[2] = -bbox3d_nuscenes[0]
      bbox3d[3] = -bbox3d_nuscenes[3]
      bbox3d[4] =  bbox3d_nuscenes[5]
      bbox3d[5] =  bbox3d_nuscenes[4]
    
    R = roty(bbox3d[3])
    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
    corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
    corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]

    return np.transpose(corners_3d)
    
def rotation_to_positive_z_angle(rotation):
    q = Quaternion(rotation)
    angle = q.angle if q.axis[2] > 0 else -q.angle
    return angle



