from __future__ import division
import argparse
import datetime
import os
import sys

import scene_ass.inout.correspondences_writer as io_cw
import scene_ass.inout.reader as r
import scene_ass.knn_associator.knn_associator as knna
import scene_ass.knn_associator.bow_trainer as bowt
import scene_ass.runner.iassociator as ia

class writible_dir_checker(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.W_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


def get_current_scripts_path():
  return os.getcwd()


def list_associator_train_factories():
  return [lambda _  = None : bow.knn_trainer()]


def list_associator_infer_factories():
  return [lambda _  = None : knna.knn_associator(0)]


class runner():
  def __init__(self, mode, dataset, output):
    self.reader = r.reader(dataset)
    self.output = output
    self.associator_factories = list_associator_train_factories \
      if mode == 'train' else list_associator_infer_factories
    self.mode = mode


  def generate_out_path(self, name):
    return self.output + '/matches_' + name + '_' + \
           datetime.datetime.now().strftime("%Y_%d_%B_%H_%M") + '.csv'


  def train_trainee(self, associator):
    assert associator.get_running_mode() == 'train' or associator.get_running_mode() == 'two_pass'
    print('{}: train alg [{}]'.format(type(self).__name__,
                                      type(associator).__name__))
    for i in range(self.reader.get_length()):
      fi = self.reader.get_frame(i)
      associator.process_frame(i, fi)
    associator.train_finish()
    print('{}: train alg [{}] .. done'.format(type(self).__name__,
                                              type(associator).__name__))


  def test_testee(self, associator):
    print('{}: test alg [{}]'.format(type(self).__name__,
                                     type(associator).__name__))
    
    if associator.get_running_mode() == 'one_pass':
      assert False, 'not impl'

    elif associator.get_running_mode() == 'two_pass':
      self.train_trainee(associator)
      return self.second_pass(associator)


  def second_pass(self, associator):
    matches = io_cw.correspondences_writer()
    prev_fraction = 0
    for i in range(self.reader.get_length()):
      curr_fraction = i / self.reader.get_length()
      if (curr_fraction > prev_fraction + 0.1):
        prev_fraction = curr_fraction
        print('{:.1f}% done'.format(curr_fraction * 100.))

      fi = self.reader.get_frame(i)
      for j in range(self.reader.get_length()):
        if i == j:
          continue
        fj = self.reader.get_frame(j)
        confidence = associator.match_frames(i, fi, j, fj)
        matches.push_back(i, j, confidence)
    print('diff {} same {}'.format(associator.cnn_classifier.count_diff, associator.cnn_classifier.count_same))
    return matches


  def process(self):
    for associator_factory in self.associator_factories():
      associator = associator_factory()
      assert isinstance(associator, ia.interface_associator)
      associator.set_interm_dir(self.output)
      if self.mode == 'test':
        matches = self.test_testee(associator)
        out_ = self.generate_out_path(type(associator).__name__)
        print('put correspondences to {}'.format(out_))
        matches.write(out_)
      elif self.mode == 'train':
        self.train_trainee(associator)


def main():
  parser = argparse.ArgumentParser('runner')
  parser.add_argument('-d', '--dataset',
                      dest = 'dataset',
                      help = 'path to dataset',
                      required = True)
  parser.add_argument('-m', '--mode',
                      dest = 'mode',
                      choices = ['train', 'test'],
                      help = 'running mode',
                      required = True)
  parser.add_argument('-o', '--output_dir',
                      dest = 'output_dir',
                      action = writible_dir_checker,
                      help = 'dir to store results',
                      default = get_current_scripts_path())

  args = parser.parse_args()
  p = runner(args.mode, args.dataset, args.output_dir)
  p.process()


if '__main__' == __name__:
  main()
