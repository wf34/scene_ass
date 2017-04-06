
import datetime
import sys

import numpy as np

import skimage.feature
import sklearn.cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import skimage.color

import cv2

import scene_ass.runner.iassociator as ia
#import scene_ass.cnn.cnn_inference as sa_ci

STEP = 66
RADIUS = 24
DESCRIPTORS_AMOUNT_ALONG_SIDE = 6


def get_time_string():
  return datetime.datetime.now().strftime("%Y_%d_%B_%H_%M")


def obtain_frame_descriptors(image):
  return skimage.feature.daisy(image, step=STEP, radius=RADIUS, rings=1, \
                               histograms=6, \
                               orientations=8, visualize=False)


def obtain_stacked_descriptors(descriptors):
    length = np.prod(descriptors.shape[0:2])
    descriptor_cardinality = descriptors.shape[2]
    descriptors = descriptors.reshape(length, descriptor_cardinality)
    return descriptors


def obtain_frame_histogram(image_descriptors, cluster_estimator):
  clusterings = cluster_estimator.predict(image_descriptors)
  frame_histogram, _ = np.histogram(clusterings, bins=np.arange(512), density = True)
  return frame_histogram


class salient_region_data:
    sift_descriptors_ = None
    sift_keypoints_ = None
    points_amount = 150
    
    def __init__(self, keypoints, descriptors):
        self.sift_keypoints_ = keypoints
        self.sift_descriptors_ = descriptors
    
    def get_descriptors(self):
        return self.sift_descriptors_
    
    def get_keypoints(self):
        return self.sift_keypoints_


def compute_salient_data(image):
    sift = cv2.xfeatures2d.SIFT_create(salient_region_data.points_amount)
    keypoints, descriptors = sift.detectAndCompute(image.astype(np.uint8), None)    
    return salient_region_data(keypoints, descriptors)


def assess_homography_on_image_pair(query_image_sal_data, sample_image_sal_data):    
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(query_image_sal_data.get_descriptors(), \
                            sample_image_sal_data.get_descriptors())

    query_keypoints = query_image_sal_data.get_keypoints()
    sample_keypoints = sample_image_sal_data.get_keypoints()
    
    query_points = np.float32([ query_keypoints[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    sample_points = np.float32([ sample_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(query_points, sample_points, cv2.RANSAC, 10.0)
    if mask is None:
      return 0.
    return np.sum(mask) / salient_region_data.points_amount

SCORE_BY_HOM = 0 
SCORE_BY_CNN = 1

class knn_associator(ia.interface_associator):
  def __init__(self, mode):
    self.interm_dir = None
    self.descriptor_cardinality_ = None
    self.cluster_estimator_ = None
    self.knn_estimator_ = None
    self.histograms = {}
    self.neighbours = {}
    self.salient_datums = {}
    self.neighbors_amount = 20
    assert mode in [SCORE_BY_HOM, SCORE_BY_CNN]
    self.score_mode = mode
    if self.score_mode == SCORE_BY_CNN:
      self.cnn_classifier = None#sa_ci.cnn_inference()
    np.random.seed(34)


  def process_frame(self, index, f):
    frame = f['grayscale']
    if not getattr(self, 'cluster_estimator_'):
      self.load_all()
    if index % 100 == 0:
      self.pprint('chewing through {}th frame...'.format(index))

    current_descriptors = obtain_frame_descriptors(frame.astype(np.uint8))
    current_descriptors = obtain_stacked_descriptors(current_descriptors)
    current_histogram = obtain_frame_histogram(current_descriptors,
                                               self.cluster_estimator_)
    self.histograms[index] = current_histogram


  def train_finish(self):
    self.pprint('fit knn ...')
    histograms_ = []
    for h in self.histograms.values():
      histograms_.append(h)
    histograms_base = np.array(histograms_)
    self.knn_estimator_ = NearestNeighbors(self.neighbors_amount, algorithm='auto').fit(histograms_base)
    p = self.interm_dir + '/knn_' + get_time_string()
    joblib.dump(self.knn_estimator_, p)
    self.pprint('trained knn classifier saved at [{}]'.format(p))


  def precompute_neighbours(self, index):
    dsts, indices = self.knn_estimator_.kneighbors(self.histograms[index].reshape(1, -1))
    d = dsts.ravel().tolist()
    i = indices.ravel().tolist()
    z = zip(i, d)
    z = list(filter(lambda x : x[0] != index and x[1] < 0.2, z))
    self.neighbours[index] = list(map(lambda x : x[0], z))
    if self.neighbours[index]:
      assert isinstance(self.neighbours[index][0], int)


  def precompute_saliency(self, index, image):
    self.salient_datums[index] = compute_salient_data(image)


  def match_frames(self, ref_index, ref_frame, query_index, query_frame):
    if not self.cluster_estimator_:
      self.load_all()
    assert self.knn_estimator_ is not None, 'required by construction'

    for i in [ref_index, query_index]:
      if i not in self.histograms:
        assert False, 'impossible by construction'
      if i not in self.neighbours:
        self.precompute_neighbours(i)

    if query_index not in self.neighbours[ref_index] and \
       ref_index not in self.neighbours[query_index]:
      return 0.
    else:
      return self.score(ref_index, ref_frame, query_index, query_frame)


  def score(self, ref_index, ref_frame, query_index, query_frame):
    if self.score_mode == SCORE_BY_HOM:
      for i, fi in zip([ref_index, query_index], [ref_frame, query_frame]):
        if i not in self.salient_datums:
          self.precompute_saliency(i, fi['grayscale'])
      return assess_homography_on_image_pair(self.salient_datums[ref_index],
                                             self.salient_datums[query_index])
    else:
      return self.cnn_classifier.do_cnn_inference(ref_frame['color'],
                                                  query_frame['color'])


  def set_interm_dir(self, interm_dir):
    self.interm_dir = interm_dir


  def load_all(self):
   assert getattr(self, 'interm_dir') is not None
   if sys.version_info > (3, 0):
     self.cluster_estimator_ = joblib.load(self.interm_dir + '/kmeans_2017_12_April_23_48.pkl')
   else:
     self.cluster_estimator_ = joblib.load(self.interm_dir + '/kmeans_2017_27_April_23_07.pkl')
   assert self.cluster_estimator_ is not None


  def get_running_mode(self):
    return 'two_pass'

