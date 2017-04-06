
import argparse
import sys
import csv
import errno    
import os
import itertools
import shutil
import random

import numpy as np
import scipy as sp
import scipy.ndimage
import scipy.misc

import scene_ass.inout.correspondences_loader as io_cl
import scene_ass.inout.scene_sequence as io_ss

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
           return True 
        else:
            raise
    return False


def get_image_distance(f1, f2):
  ffs = []
  #print('frame shape', f1.shape)
  for f in [f1, f2]:
    ffs.append(f.astype(float) / 255)

  diff = ffs[0] - ffs[1]
  for ch, chn in enumerate(['r', 'g', 'b']):
    diff_ch_range = np.max(diff[:, :, ch]) - np.min(diff[:, :, ch])
    diff[:, :, ch] = diff[:, :, ch] / diff_ch_range
    diff[:, :, ch] = diff[:, :, ch] - np.min(diff[:, :, ch])
    #for fi, f in enumerate(ffs + [diff]):
    #  print('frame#{} channel {} range ({}..{})'.format(fi, chn, np.min(f[:, :, ch]), np.max(f[:, :, ch])))

  diff *= 255.  
  return diff.astype(np.uint8)


class dataset_creator_monitor:
  POSITIVE = 1
  NEGATIVE = 0
  def __init__(self, input_dir, output_dir):
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.images = {} # id -> ndimage
    self.train_file = open(os.path.join(self.output_dir, 'train.txt'), 'w')
    self.writer = csv.writer(self.train_file, delimiter = ' ')

  def create_sample(self, pair, sample_type):
    for image_id in pair:
      if not image_id in self.images:
        self.images[image_id] = \
          sp.ndimage.imread(os.path.join(self.input_dir,
                                         '{:04d}.png'.format(image_id)))
    diff = get_image_distance(self.images[pair[0]], self.images[pair[1]])
    #diff_ = sp.misc.imresize(diff, (diff.shape[0] // 15, diff.shape[1] // 15, diff.shape[2]))
    #np.set_printoptions(threshold=np.nan)
    #for i in range(3):
    #  print(i, '\n', diff_[:, :, i])
    #diff = self.images[pair[0]] / 2 + self.images[pair[1]] / 2
    filename = os.path.join(os.path.abspath(self.output_dir),
                                   '{:04d}_{:04d}.png'.format(pair[0],
                                                              pair[1]))
    sp.misc.imsave(filename, diff)
    #sp.misc.imsave(os.path.join(os.path.abspath(self.output_dir), '{:04d}.png'.format(pair[0])), self.images[pair[0]])
    #sp.misc.imsave(os.path.join(os.path.abspath(self.output_dir), '{:04d}.png'.format(pair[1])), self.images[pair[1]])
    self.writer.writerow([filename, sample_type])


def main():
  parser = argparse.ArgumentParser('train_prep')
  parser.add_argument('-i', '--input_dir',
                      dest = 'input_dir',
                      help = 'path to dataset',
                      required = True)
  parser.add_argument('-g', '--ground_truth',
                      dest = 'ground_truth',
                      help = 'path to ground truth',
                      required = True)
  parser.add_argument('-o', '--output_dir',
                      dest = 'output_dir',
                      help = 'dir to store cnn dataset',
                      required = True)
  args = parser.parse_args()
  exists = mkdir_p(args.output_dir)
  if exists:
    shutil.rmtree(args.output_dir)
    mkdir_p(args.output_dir)

  correspondences_gt = io_cl.read_ground_truth(args.ground_truth)
  gt_scenes = io_ss.scene_sequence.create_from_ground_truth(correspondences_gt)

  dc = dataset_creator_monitor(args.input_dir, args.output_dir)
  all_frames = set(range(gt_scenes.get_max_frame_index() + 1))
  overall_samples = 0
  for gt_scene in sorted(gt_scenes.scenes, key = lambda x : len(x)):
    print('so far {} samples; doing pos for curr scene (card = {}) ...'.format(overall_samples, len(gt_scene)))
    positive_samples_in_scene = 0
    for pair in itertools.combinations(gt_scene, 2):
      if pair[0] == pair[1]:
        continue
      dc.create_sample(pair, dataset_creator_monitor.POSITIVE)
      positive_samples_in_scene += 1
    print('got {} pos samples; do negatives'.format(positive_samples_in_scene))

    outside_of_scene_frames = list(all_frames - set(gt_scene))
    gt_scene_l = list(gt_scene)
    negative_samples_in_scene = 0
    for _ in range(positive_samples_in_scene * 3):
       j = random.choice(outside_of_scene_frames)
       i = random.choice(gt_scene_l)
       dc.create_sample([i, j], dataset_creator_monitor.NEGATIVE)
       negative_samples_in_scene += 1
    overall_samples += (positive_samples_in_scene + negative_samples_in_scene)
   
  

if '__main__' == __name__:
  if sys.version_info > (3, 0):
    # print('scipy', scipy.__version__)
    random.seed(34)
    main()
  else:
    print('app was tested only with python3')

