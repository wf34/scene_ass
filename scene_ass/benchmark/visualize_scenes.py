
import argparse
import sys
import os

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as mfonts

import scene_ass.inout.correspondences_loader as io_cl
import scene_ass.inout.scene_sequence as io_ss
import scene_ass.benchmark.color_generator as cg


# max in container of containers
def get_max_index(sequence):
  return max(map(lambda x : max(x), sequence))


def make_visualization(scene_sequence, canvas, sufficient_scene_size):
  s = sorted(scene_sequence.scenes, key = lambda x : len(x), reverse = True)
  s = list(filter(lambda x : len(x) >= sufficient_scene_size, s))
  assert len(s) > 0, 'there are no scenes to match required size'

  max_frame = get_max_index(s)
  scenes_amount = len(s)
  print('will consider {} significant scenes of min size {}'.format( \
    scenes_amount, sufficient_scene_size))
  s = sorted(s, key = lambda x : min(x))
  colgen = cg.color_generator(scenes_amount)
  ROWS = 50
  BORDER = 0
  BAR_WIDTH = 1
  CHANNELS = 3
  bars = np.zeros((ROWS, max_frame * (BAR_WIDTH + BORDER), CHANNELS))
  scene_patches = []
  for i, scene in enumerate(s):
    scene_color = colgen.get_next_color()
    for frame in scene:
      bars[:, frame*BAR_WIDTH:(frame+1)*BAR_WIDTH, :] = scene_color

    scene_patch = mpatches.Patch(color = scene_color, \
        label='{}th scene of size {}'.format(i, len(scene)))
    scene_patches.append(scene_patch)

  font_props = mfonts.FontProperties()
  font_props.set_size('small')
  canvas.legend(handles = scene_patches, prop = font_props, \
                loc='upper center', bbox_to_anchor=(0.5, -0.8),
                ncol = 3)
  canvas.axes.get_yaxis().set_visible(False)
  canvas.imshow(bars)


def main():
  parser = argparse.ArgumentParser('Visualize Scenes')
  parser.add_argument('-c', '--correspondences_path',
                      dest = 'correspondences_path',
                      help = 'path to scene data',
                      required = True)
  parser.add_argument('-b', '--backend',
                      dest = 'backend',
                      help = 'path to ground truth',
                      choices = ['gui', 'png'],
                      default = 'gui',
                      required = False)
  parser.add_argument('-s', '--scene_size',
                      dest = 'sufficient_scene_size',
                      help = 'sufficient scene size for visualization',
                      type = int,
                      default = 15,
                      required = False)
  parser.add_argument('-t', '--threshold',
                      dest = 'threshold',
                      help = 'active for matches',
                      type = float,
                      default = .95,
                      required = False)
  parser.add_argument('-o', '--output',
                      dest = 'output',
                      help = 'path to output image',
                      default = os.getcwd() + '/visualize_scenes.png',
                      required = False)

  args = parser.parse_args()
  if args.backend == 'png':
    assert args.output is not None and \
           isinstance(args.output, str) and \
           args.output[-4:] == '.png'

  scene_sequence = io_ss.scene_sequence.create( \
      args.correspondences_path, args.threshold)
  assert len(scene_sequence.scenes) > 0

  f = plt.figure()
  ax1 = f.add_subplot(111)
  make_visualization(scene_sequence, ax1, args.sufficient_scene_size)
  
  if args.backend == 'gui':
    f.show()
    plt.show()
  elif args.backend == 'png':
    f.savefig(args.output)

if '__main__' == __name__:
  if sys.version_info > (3, 0):
    main()
  else:
    print('app was tested only with python3')
