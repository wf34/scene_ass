from __future__ import division
import os
import scipy as sp
import scipy.ndimage

class reader():
  def __init__(self, path, mode = 'preload'):
    assert not path == ''
    assert os.path.isdir(path), 'something wrong with path {}'.format(path)
    assert mode == 'preload' or mode == 'onfly'
    self.dataset_path = path
    self.mode = mode
    self.frames = []
    self.frames_data = None
    self.index_all_frames(self.dataset_path)
    self.preload()
    

  def read_from_fs(self, path):
    return {'grayscale' : sp.ndimage.imread(path, flatten = True),
            'color' : sp.ndimage.imread(path)}


  def preload(self):
    if not self.mode == 'preload':
      return
    print('{} preloads dataset ..'.format(type(self).__name__))
    self.frames_data = []
    prev_fraction = 0.
    for i in range(self.get_length()):
      curr_fraction = i / self.get_length()
      if curr_fraction > prev_fraction + .25:
        prev_fraction = curr_fraction
        print('{:.1f}% done'.format(curr_fraction * 100.))
      self.frames_data.append(self.read_from_fs(self.frames[i]))
    print('{} preload done'.format(type(self).__name__))
      


  def index_all_frames(self, path):
    fullpathframes = map(lambda x : os.path.join(path, x), os.listdir(path))
    frames = filter(lambda x : os.path.isfile(x) and x[-4:] == '.png', fullpathframes)
    self.frames = sorted(frames)


  def get_length(self):
    assert len(self.frames) > 0, 'reader not inited'
    return len(self.frames)


  def get_frame(self, i):
    assert 0 <= i and i < self.get_length(), \
           'ask {} when limits [{}, {})'.format(i, 0, self.get_length())
    if self.mode == 'preload':
      return self.frames_data[i] 
    else:
      return self.read_from_fs(self.frames[i])

