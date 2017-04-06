import os
import shutil

import numpy as np
import scipy as sp
import scipy.misc

os.environ["GLOG_minloglevel"] = "1"
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

import scene_ass.cnn.train_prep as sa_tp

caffe_root = '/home/dmitri/projects/caffe/'
infer_weigths = '/home/dmitri/projects/graphicon_paper/scene_ass/cnn/weights.pretrained.caffemodel4'
model = '/home/dmitri/projects/graphicon_paper/scene_ass/cnn/alex_deploy'

class cnn_inference():
  def __init__(self):
    self.net = caffe.Classifier(model,
                            infer_weigths,
                            mean = \
                            np.load(caffe_root + 'data/ilsvrc12/scene_ass_mean.npy').mean(1).mean(1),
                            channel_swap=(2,1,0),
                            raw_scale=255,
                            image_dims=(227, 227))
    self.counter = 0
    self.count_same = 0
    self.count_diff = 0
    if os.path.exists('/tmp/1'):
      shutil.rmtree('/tmp/1')
    os.makedirs('/tmp/1')

  def do_cnn_inference(self, f1, f2):
    diff = sa_tp.get_image_distance(f1, f2)
    scipy.misc.imsave('/tmp/1/{}.png'.format(self.counter), diff)
    predictions = self.net.predict([caffe.io.load_image('/tmp/1/{}.png'.format(self.counter))])[0]
    self.counter += 1
    if 0 == np.argmax(predictions):
      self.count_diff += 1
    else:
      self.count_same += 1
    return float(predictions[1])
