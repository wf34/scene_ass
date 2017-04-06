import argparse
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import scene_ass.inout.correspondences_loader as io_cl

def main():
  parser = argparse.ArgumentParser('scores_distrib')
  parser.add_argument('-m', '--matches',
                      dest = 'matches',
                      nargs = '+',
                      help = 'path to matches csv file',
                      required = True)

  args = parser.parse_args()
  for matches in args.matches:
    correspondences = io_cl.read_matches(matches)
    scores = list(map(lambda x : x[2], correspondences))
    scores = sorted(scores)
    print(scores[:10])
    plt.plot(scores)
  plt.show()


if '__main__' == __name__:
  if sys.version_info > (3, 0):
    main()
  else:
    print('app was tested only with python3')

