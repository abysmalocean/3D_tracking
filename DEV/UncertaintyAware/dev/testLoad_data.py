from loadData import raw


# Change this to the directory where you store KITTI data
basedir = '../data_test/'

# Specify the dataset to load
date = '2011_09_26'
drive = '0015'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.raw(basedir, date, drive)
dataset = raw(basedir, date, drive, frames=range(0, 20, 5))