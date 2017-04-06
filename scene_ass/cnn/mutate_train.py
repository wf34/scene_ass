import sys
import csv
import random

with open(sys.argv[1], 'r') as inp_file, open(sys.argv[2], 'w') as out_file:
  inp_file.readline()
  r = csv.reader(inp_file, delimiter = ' ')
  w = csv.writer(out_file, delimiter = ' ')
  rows = []
  for row in r:
    row = [row[0], int(row[1])]
    rows.append(row)
  
  random.shuffle(rows)
  for row in rows:
    w.writerow(row)
