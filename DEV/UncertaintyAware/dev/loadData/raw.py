"""Provides 'raw', which loads and parses raw KITTI data."""

import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np

import loadData.utils as utils



class raw:
    """Load and parse raw data into a usable format."""

    def __init__(self, base_path, date, drive, **kwargs):
        """Set the path and pre-load calibration data and timestamps."""
        self.dataset = kwargs.get('dataset', 'extract')
        self.drive = date + '_drive_' + drive + '_' + self.dataset
        self.calib_path = os.path.join(base_path, date)
        self.data_path = os.path.join(base_path, date, self.drive)
        self.frames = kwargs.get('frames', None)
        self.oxts_files_full = []
        self.timestamps_second = []
        self.w = 1.60
        self.h = 2.71 
        self.offset = [0.32, 0.05]

        self.afterSmooth = None

        
        self.oxts_files = []


        # Default image file extension is '.png'
        #self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists(self.frames)
        # Pre-load data that isn't returned as a generator
        self._load_timestamps()
        self.convert_dt_tosecond()
        self._load_oxts()

        if self.frames == None:
            self.begin_index = None
            self.end_index   = None
            self.get_begin_end_index(self.frames)
        

        ## extracted Measurements
        self.heading = []
        self.locs    = []
        for item in self.oxts:
            self.heading.append(item.heading)
            self.locs.append(item.Trans)
        self.heading = np.array(self.heading)
        self.locs    = np.array(self.locs)
        # for ground truth Data generation
        # use high frequency smoothed method. 
        
        # 1. Measurement Sigma
        self.GPS_x = 0.02
        self.GPS_y = 0.02
        self.GPS_h = 0.01 

    def add_some_noise(self, sigma_x = 1.0, sigma_y= 1.0, sigma_theta = 0.1):
        self.GPS_x = sigma_x
        self.GPS_y = sigma_y
        self.GPS_h = sigma_theta
        n = len(self.oxts)
        noise_x = np.random.normal(0, sigma_x     , n)
        noise_y = np.random.normal(0, sigma_y     , n)
        noise_t = np.random.normal(0, sigma_theta , n)
        
        # Noise Measurements
        self.heading   += noise_t
        self.locs[:,0] += noise_x 
        self.locs[:,1] += noise_y


    def convert_dt_tosecond(self):
        for item in self.timestamps: 
            second = (item - self.timestamps[0]).total_seconds()
            self.timestamps_second.append(second)

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.timestamps)

    def get_begin_end_index(self, indices): 
        if indices == None: 
            self.begin_index = 0
            self.end_index   = len(self.oxts)
            return
        indices = [i for i in indices]
        self.begin_index = indices[0]
        self.end_index   = indices[-1]

    def _get_file_lists(self, indices):
        """Find and list data files for each sensor."""
        self.oxts_files = sorted(glob.glob(
            os.path.join(self.data_path, 'oxts', 'data', '*.txt')))

        # Subselect the chosen range of frames, if any
        if indices is not None:
            self.oxts_files = utils.subselect_files(
                self.oxts_files, indices)

    def _load_timestamps(self):
        """Load timestamps from file."""
        timestamp_file = os.path.join(
            self.data_path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps.append(t)

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.timestamps = [self.timestamps[i] for i in self.frames]

    def _load_oxts(self):
        """Load OXTS data from file."""
        self.oxts = utils.load_oxts_packets_and_poses(self.oxts_files)
