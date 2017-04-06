import sys
import csv
with open(sys.argv[1], 'r') as inp_file, open(sys.argv[2], 'w') as out_file:
  inp_file.readline()
  r = csv.reader(inp_file, delimiter = ' ')
  w = csv.writer(out_file, delimiter = ' ')
  for row in r:
    row = list(map(lambda x: int(x), row))
    if row[0] > 216 or row[1] > 216:
      continue
    w.writerow(row)
