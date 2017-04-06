import enum
import tarjan

import scene_ass.inout.correspondences_loader as io_cl

def init_cell(dict_, key):
  if not key in dict_:
    dict_[key] = []
  return dict_


def convert_graph_to_clique_list(graph):
  cliques = tarjan.tarjan(graph)
  cliques_ = []
  for clique in cliques:
      cliques_.append(set(clique))
  return cliques_

class match_status(enum.Enum):
  TRUE_POSITIVE = 1 # registered on the same scene in gt and inferred, given specific overlap criteria
  TRUE_NEGATIVE = 2 # not part of any scenes in gt and inferred
  FALSE_POSITIVE = 3 # matched in inferred to scene not sufficiently overlapping with gt, or gt non-matching this frame at all
  FALSE_NEGATIVE = 4 # not matched in inferred but matched in gt
  

class scene_sequence():
  sure_confidence = 1.
  trivial_threshold = -1

  # class to describe and encapsulate ground_truth or estimated scene sequence
  def __init__(self, correspondences_type, correspondences, threshold = None):
    assert 'matches' == correspondences_type or \
           'ground_truth' == correspondences_type
    self.type = correspondences_type

    assert len(correspondences) > 0
    proper_cols_amount = 2 if self.is_gt() else 3
    assert len(correspondences[0]) == proper_cols_amount, \
      'proper_cols_amount = {}, correspondences_type = {} row: {}'.format(
      proper_cols_amount, correspondences_type, correspondences[0])

    self.correspondences = correspondences
    self.build_scenes(threshold)


  def get_type(self):
    return self.type


  def get_max_frame_index(self):
    findices = list(map(lambda x : x[0], self.correspondences))
    return max(findices)

  
  def is_gt(self):
    return self.get_type() == 'ground_truth'


  def get_scene(self, frame_index):
    assert isinstance(self.scenes, list)
    assert len(self.scenes) > 0 or not self.is_gt()
    for sc in self.scenes:
      if frame_index in sc:
        return sc
    return None


  def build_scenes(self, threshold):
    if self.is_gt():
      assert threshold == None
      threshold = scene_sequence.trivial_threshold
    else:
      assert isinstance(threshold, float) and 0 < threshold and threshold < 1, \
        'Thresh most be float and in (0, 1) ~ {:.6f}'.format(threshold)

    connectivity_graph = {}
    for row in self.correspondences:
      first_id = row[0]
      second_id = row[1]
      confidence = self.sure_confidence if self.is_gt() else row[2] 

      if confidence > threshold:
        connectivity_graph = init_cell(connectivity_graph, first_id)
        connectivity_graph = init_cell(connectivity_graph, second_id)

        connectivity_graph[first_id].append(second_id)
        connectivity_graph[second_id].append(first_id)

    self.scenes = convert_graph_to_clique_list(connectivity_graph)


  def create_from_matches(correspondences, threshold):
    return scene_sequence('matches', correspondences, threshold)


  def create_from_ground_truth(correspondences):
    return scene_sequence('ground_truth', correspondences)


  def create(path, threshold = None):
    c_type = io_cl.infer_correspondences_type(path)
    c = io_cl.read_correspondences(path)
    assert c != None
    return scene_sequence.create_from_matches(c, threshold) \
      if c_type == 'matches' else \
      scene_sequence.create_from_ground_truth(c)


  def check_scene_overlap(frame_index,
                          inferred_scene_sequence,
                          gt_scene_sequence,
                          required_overlap = .7):
    assert isinstance(inferred_scene_sequence, scene_sequence) and \
           inferred_scene_sequence.get_type() == 'matches'
    assert isinstance(gt_scene_sequence, scene_sequence) and \
           gt_scene_sequence.get_type() == 'ground_truth'

    inferred_scene = inferred_scene_sequence.get_scene(frame_index)
    gt_scene = gt_scene_sequence.get_scene(frame_index)

    if inferred_scene == None and gt_scene == None:
      return match_status.TRUE_NEGATIVE
    if inferred_scene == None and gt_scene != None:
      return match_status.FALSE_NEGATIVE
    if inferred_scene != None and gt_scene == None:
      return match_status.FALSE_POSITIVE
    
    assert isinstance(inferred_scene, set) and len(inferred_scene) > 0
    assert isinstance(gt_scene, set) and len(gt_scene) > 0

    cardinality_of_bigger_scene = max([len(inferred_scene), len(gt_scene)])
    overlapping_scene = inferred_scene.intersection(gt_scene)
    overlap = len(overlapping_scene) / cardinality_of_bigger_scene
    return match_status.TRUE_POSITIVE if overlap > required_overlap else \
           match_status.FALSE_POSITIVE

