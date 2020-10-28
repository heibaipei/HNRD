import xlwt
import numpy as np

# drug_name = np.loadtxt('DrugsName')

workbook = xlwt.Workbook(encoding = 'utf-8')

worksheet = workbook.add_sheet('My Worksheet')

f = open('DrugsName')
s =1
for line in f.readlines():
    a = line.strip()
    worksheet.write(s, 0, label = a)
    s = s+1

f = open('DiseasesName')
s =1
for line in f.readlines():
    a = line.strip()
    worksheet.write(s, 4, label = a)
    s = s+1
workbook.save('Excel_test.xls')