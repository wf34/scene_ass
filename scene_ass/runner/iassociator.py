
class interface_associator():
  # where to write to at train and where to load from at test
  def set_interm_dir(self, interm_dir):
    raise NotImplementedError()

  def pprint(self, *args):
    print('{}: '.format(type(self).__name__) + ' '.join(map(str,args)))

  # function for train
  def process_frame(self, index, frame):
    raise NotImplementedError()

  # returns confidence score [0, 1)
  def match_frames(self, ref_index, ref_frame, query_index, query_frame):
    raise NotImplementedError()

  # post train
  def train_finish(self, index, frame):
    raise NotImplementedError()

  # where to write to at train and where to load from at test
  def set_interm_dir(self, interm_dir):
    raise NotImplementedError()

  # returns mode:  'one_pass' or 'two_pass' or 'train'
  # one pass will be called with match_frames
  # two pass will be first "trained" by process frame,
  #     then called
  # train - process all frames than exit
  def get_running_mode(self):
      raise NotImplementedError()
