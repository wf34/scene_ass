
import numpy as np

import sklearn.cluster
from sklearn.externals import joblib

import scene_ass.runner.iassociator as ia
import scene_ass.knn_associator.knn_associator as knna

class bow_trainer(ia.interface_associator):
  def __init__(self):
    self.interm_dir = None


  def process_frame(self, index, frame):
      current_descriptors = knna.obtain_frame_descriptors(frame.astype(np.uint8))
      current_descriptors = knna.obtain_stacked_descriptors(current_descriptors)
      if index == 0:
        self.all_descriptors_ = []
        self.descriptor_cardinality_ = current_descriptors.shape[1]
      if index % 100 == 0:
        print('{}: chewing through {}th frame...'.format(type(self).__name__,
                                                         index))
      self.all_descriptors_.append(current_descriptors)

  def train_finish(self):
   assert self.interm_dir is not None 
   assert self.all_descriptors_ is not None
   self.all_descriptors_ = np.array(self.all_descriptors_)
   self.all_descriptors_ = knna.obtain_stacked_descriptors(self.all_descriptors_)
   assert len(self.all_descriptors_.shape) == 2 and \
          self.all_descriptors_.shape[1] == self.descriptor_cardinality_, self.all_descriptors_.shape

   print('{}: start kmeans fit'.format(type(self).__name__))
   self.cluster_estimator_ = sklearn.cluster.KMeans(n_clusters = 512,
                                                    precompute_distances=True,
                                                    n_jobs=-2)
   self.cluster_estimator_.fit(self.all_descriptors_)
   p = self.interm_dir + '/kmeans_' + knna.get_time_string() + '.pkl'
   joblib.dump(self.cluster_estimator_, p)
   print('{}: trained kmeans classifier saved at [{}]'.format(type(self).__name__, p))


  def get_running_mode(self):
    return 'train'


  def set_interm_dir(self, interm_dir):
    self.interm_dir = interm_dir
