"""Provides helper methods for loading and parsing KITTI data."""

from collections import namedtuple
import numpy as np

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple("OxtsData", 'packet, Trans, R, heading')

def subselect_files(files, indices):
    try:
        files = [files[i] for i in indices]
    except:
        pass
    return files

def load_oxts_packets_and_poses(oxts_files): 
    """
        Generator tp read OXTS ground truth data.

        Poses are given in an East-North-Up coordinate system
        whose origina is the first GPS position. 
    """
    # Scale for Mercator projection(from the lat value)
    scale = None
    # Origin of the global coordinate system(First GPS position)
    origin = None

    oxts = []

    for filename in oxts_files: 
        with open(filename, 'r') as f: 
            for line in f.readlines():
                line = line.split()
                # Last file entries are flags and counts so they are intiger
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)
                
                if scale is None: 
                    scale = np.cos(packet.lat * np.pi / 180.)
                R, t, heading = pose_from_oxts_packet(packet, scale)

                if origin is None: 
                    origin = t
                oxts.append(OxtsData(packet, t-origin, R, heading))
    return oxts

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def pose_from_oxts_packet(packet, scale): 
    """
    Helper Method to compute a SE(3) pose matrix from an OXTS packet
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    heading = packet.yaw

    return R, t, heading




