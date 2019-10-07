import cv2
import numpy as np
import glob

table_path = 'table/'
linear_path = 'linear/'
dqn_path = 'dqn2/20181112230948/'

table_files = glob.glob(table_path + 'fig/' + 'mountain*')
linear_files = glob.glob(linear_path + 'fig/' + 'mountain*')
dqn_files = glob.glob(dqn_path + 'fig/' + 'mountain*')

assert len(table_files) == len(linear_files)

## 2 figure concat
#for i in range(len(table_files)):
#    file_name = table_files[i].split('/')[-1]
#    print(file_name)
#    im1 = cv2.imread(table_files[i])
#    im2 = cv2.imread(linear_files[i])
#    im_h = cv2.hconcat([im1,im2])
#    cv2.imwrite('imh_' + file_name, im_h)

# 3 figure concat
for i in range(len(table_files)):
    file_name = table_files[i].split('/')[-1]
    print(file_name)
    im1 = cv2.imread(table_files[i])
    im2 = cv2.imread(linear_files[i])
    im3 = cv2.imread(dqn_files[i])
    im_h = cv2.hconcat([im1,im2, im3])
    cv2.imwrite('imh_' + file_name, im_h)

