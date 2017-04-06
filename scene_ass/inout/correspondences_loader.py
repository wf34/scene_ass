import csv
import os

def infer_correspondences_type(path):
  with open(path, 'r') as the_file:
    reader = csv.reader(the_file, delimiter=' ')
    row = next(reader)
    return 'matches' if len(row) == 3 else 'ground_truth'

def read_correnspondences_csv(path, csv_type):
  if 'matches' != csv_type and 'ground_truth' != csv_type:
    # matches supposed to have query_id,train_id,score
    # and gt supposed to have query_id,train_id
    raise ValueError('unexpected csv type')

  assert os.path.isfile(path)
  correnspondences = []
  with open(path, 'r') as the_file:
    reader = csv.reader(the_file, delimiter=' ')
    is_first = True
    for row in reader:
      try:
        if 'matches' == csv_type:
            row = list(map(lambda x : int(x), row[:-1])) + [float(row[-1])]
        elif 'ground_truth' == csv_type: 
            row = list(map(lambda x : int(x), row))
      except ValueError:
        if is_first:
          # everyhing is fine, it was header
          is_first = False
          continue
        else:
          raise
      correnspondences.append(row)
  return correnspondences


def read_ground_truth(path):
  return read_correnspondences_csv(path, 'ground_truth')


def read_matches(path):
  return read_correnspondences_csv(path, 'matches')

def read_correspondences(path):
  if 'matches' == infer_correspondences_type(path):
    return read_matches(path)
  else:
    return read_ground_truth(path)


