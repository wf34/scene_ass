import argparse
import os
import csv

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

linestyles = ['-', '--', '-.', 'steps']

class line_marker_provider():
  def __init__(self):
    self.line_style = 0
    self.counter = 0

  def get_next(self):
    current_l = self.counter % len(linestyles)
    self.counter += 1
    return linestyles[current_l]


def get_roc_names(filenames):
  assert isinstance(filenames, list)
  res_filenames = []
  for filename_entry in filenames:
    if os.path.isdir(filename_entry):
      files = list(filter(lambda x : '.csv' ==  os.path.splitext(x)[1],
                         os.listdir(filename_entry)))
      res_filenames += \
        list(map(lambda x: os.path.join(filename_entry, x), files))
    else:
      res_filenames.append(filename_entry)
  return res_filenames


def get_label(filepath):
  return os.path.splitext(os.path.basename(filepath))[0]


def read_roc(roc_filename):
  assert os.path.isfile(roc_filename) and roc_filename.endswith('csv'), roc_filename
  roc = [[], []]
  with open(roc_filename, 'r') as the_file:
    csv_reader = csv.reader(the_file, delimiter = ' ')
    for row in csv_reader:
      assert len(row) == 2, row
      r = list(map(lambda x : float(x), row))
      for col in range(2):
        roc[col].append(r[col])
  assert len(roc[0]) > 0 and len(roc[0]) == len(roc[1])
  return roc


def get_mean_roc(rocs_filenames):
  rocs = []
  for rf in rocs_filenames:
    roc = read_roc(rf)
    rocs_arr = np.zeros((2, len(roc[0])))
    for i in range(2):
      rocs_arr[i, :] = np.array(roc[i])
    rocs.append(rocs_arr)

  upscale_tuple = [len(rocs)] + list(rocs[0].shape)
  rocs_ = np.zeros(upscale_tuple)
  for ri, r in enumerate(rocs):
    rocs_[ri, :, :] = r

  mean_roc_ = np.mean(rocs_, axis = 0)
  mean_roc = [mean_roc_[0, :].tolist(), mean_roc_[1, :].tolist()]
  return mean_roc


def visualize(baseline_rocfiles, advanced_rocfiles, savefile):
  f = plt.figure()
  ax = f.add_subplot(111)
  cl = [0.6, 0.]
  for i, (sort_of_rocfiles, sort_name) in enumerate(zip([baseline_rocfiles, advanced_rocfiles], ['hom', 'cnn'])):
    prov = line_marker_provider()
    for rocfile in sort_of_rocfiles:
      label = get_label(rocfile)
      roc = read_roc(rocfile)
      style = prov.get_next()
      ax.plot(roc[0], roc[1], linestyle = style, color = (cl[i], cl[i], cl[i]), label = label)
    if len(sort_of_rocfiles) == 0:
      continue
    mean_roc = get_mean_roc(sort_of_rocfiles)
    style = prov.get_next()
    ax.plot(mean_roc[0], mean_roc[1],  linewidth = 2, linestyle = style, color = (cl[i], cl[i], cl[i]), label = sort_name + '_mean')


  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=2)
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.set_xlim([0.0, 1.05])
  ax.set_ylim([0.0, 1.05])
  ax.grid(True)
  f.savefig(savefile, bbox_inches="tight")



def main():
  parser = argparse.ArgumentParser('Show in one axis')
  parser.add_argument('-b',
                      dest = 'baseline_rocs',
                      nargs = '+',
                      help = 'space-separated list of baseline roc.csv files or folder',
                      default = [])
  parser.add_argument('-a',
                      dest = 'advanced_rocs',
                      nargs = '+',
                      help = 'space-separated list of advanced roc.csv files or folder',
                      default = [])
  parser.add_argument('-o', '--output',
                      dest = 'output',
                      help = 'path to output image',
                      default = os.getcwd() + '/all_on_axis.png',
                      required = False)
  args = parser.parse_args()
  baseline_rocs = get_roc_names(args.baseline_rocs)
  advanced_rocs = get_roc_names(args.advanced_rocs)
  visualize(baseline_rocs, advanced_rocs, args.output)

if '__main__' == __name__:
  main()
