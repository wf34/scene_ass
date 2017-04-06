
import csv
import os

class correspondences_writer():

  def __init__(self):
    self.header = ['query_id', 'train_id', 'score']
    self.matches = []


  def push_back(self, query_id, train_id, score):
    assert isinstance(query_id, int)
    assert isinstance(train_id, int)
    assert isinstance(score, float)
    self.matches.append([query_id, train_id, score])


  def write(self, path):
    assert os.path.isdir(os.path.dirname(path))
    with open(path, 'w') as the_file:
      writer = csv.writer(the_file, delimiter=' ')
      writer.writerow(self.header)
      for row in self.matches:
        row = list(map(lambda x : '{:.6f}'.format(x) \
                         if isinstance(x, float) else x,\
                       row))
        writer.writerow(row)
