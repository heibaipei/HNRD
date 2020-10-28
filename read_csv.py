
import csv

with open('drug links.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)

    print(headers)
