
import argparse
import csv
import os
import sys

import numpy as np
import numpy.linalg
import sklearn.metrics

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import scene_ass.inout.correspondences_loader as io_cl
import scene_ass.inout.scene_sequence as io_ss

eps = sys.float_info.epsilon


def get_inited_counters():
  status_counters = {}
  for s in list(io_ss.match_status):
    status_counters[s] = 0
  return status_counters


def benchmark(correspondences,
              correspondences_gt,
              output_path):
  THRESHOLD_LEVELS_AMOUNT = 14
  PRINT_THRESHOLD_EVERY = 20
  
  scores = list(map(lambda x : x[2], correspondences))
  smin, smax = max(eps, min(scores)), min(1-eps, max(scores))
  assert smin > 0. and smax < 1. 
  thresholds = np.linspace(smin, smax, THRESHOLD_LEVELS_AMOUNT)

  gt_scenes = io_ss.scene_sequence.create_from_ground_truth(correspondences_gt)

  roc_x = []
  roc_y = []

  threshold_x = []
  threshold_y = []
  threshold_label = []
  last_shown_point = np.array([eps, eps]) 
  for ti, t in enumerate(thresholds):
    inferred_scenes = io_ss.scene_sequence.create_from_matches(correspondences, t)
    status_counters = get_inited_counters()     
    for fi in range(inferred_scenes.get_max_frame_index() + 1):
      status = io_ss.scene_sequence.check_scene_overlap(fi,
                                                        inferred_scenes,
                                                        gt_scenes,
                                                        .5)
      status_counters[status] += 1
    
    try:
      precision = status_counters[io_ss.match_status.TRUE_POSITIVE] / \
                  (status_counters[io_ss.match_status.TRUE_POSITIVE] + \
                   status_counters[io_ss.match_status.FALSE_POSITIVE])
    except ZeroDivisionError:
      precision = 0.

    try:
      recall = status_counters[io_ss.match_status.TRUE_POSITIVE] / \
                  (status_counters[io_ss.match_status.TRUE_POSITIVE] + \
                   status_counters[io_ss.match_status.FALSE_NEGATIVE])
    except ZeroDivisionError:
      recall = 0.

    print('statuses at: #{}, t={:.3f} TP {}, FP {}, TN {}, FN {} r {:.3f} p {:.3f}'.format(
        ti, t,
        status_counters[io_ss.match_status.TRUE_POSITIVE],
        status_counters[io_ss.match_status.FALSE_POSITIVE],
        status_counters[io_ss.match_status.TRUE_NEGATIVE],
        status_counters[io_ss.match_status.FALSE_NEGATIVE],
        precision, recall))

    roc_x.append(recall)
    roc_y.append(precision)

    current_point = np.array([recall, precision])
    if np.linalg.norm(last_shown_point - current_point, 2) > 0.15:
      last_shown_point = current_point
      threshold_x.append(recall)
      threshold_y.append(precision)
      threshold_label.append('t = {:.3f}'.format(t))

  auc_score = sklearn.metrics.auc(roc_x, roc_y, reorder = True)
  f = plt.figure()
  ax1 = f.add_subplot(111)
  ax1.plot(roc_x, roc_y)

  for label, x, y in zip(threshold_label,
                         threshold_x,
                         threshold_y):
    plt.annotate(
        label, 
        xy = (x, y),
        xytext = (40, 20),
        textcoords = 'offset points',
        ha = 'right',
        va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.2),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

  ax1.set_xlabel('Recall')
  ax1.set_ylabel('Precision')
  ax1.set_xlim([0.0, 1.05])
  ax1.set_ylim([0.0, 1.05])
  ax1.text(0.9, 0.9, 'AUC = {0:1.3f}'.format(auc_score),
           horizontalalignment='center',
           verticalalignment='center',
           transform = ax1.transAxes)
  f.savefig(output_path)
  with open(output_path + '.csv', 'w') as the_file:
    csv_writer = csv.writer(the_file, delimiter = ' ')
    for recall, precision in zip(roc_x, roc_y):
      csv_writer.writerow([recall, precision])
  return auc_score


def main():
  parser = argparse.ArgumentParser('Fair Benchmark',
    description = 'compute stats for scene matching task')
  parser.add_argument('-m', '--matches',
                      dest = 'matches',
                      help = 'path to matches csv file',
                      required = True)
  parser.add_argument('-g', '--ground_truth',
                      dest = 'ground_truth',
                      help = 'path to ground truth',
                      required = True)
  parser.add_argument('-o', '--output',
                      dest = 'output',
                      help = 'path to destination.png',
                      default = os.getcwd() + '/fair_benchmark.png',
                      required = False)

  args = parser.parse_args()
  correspondences = io_cl.read_matches(args.matches)
  correspondences_gt = io_cl.read_ground_truth(args.ground_truth)

  auc = benchmark(correspondences,
                  correspondences_gt,
                  args.output)
  print('AUC = {:.3f}'.format(auc))


if '__main__' == __name__:
  if sys.version_info > (3, 0):
    main()
  else:
    print('app was tested only with python3')

