import os
import csv
import numpy as np
import scipy.io as sio
import nilearn
from nilearn import connectome

label_file = '/Users/celery/Research/dataset/ABCD/abcd_labels_cleaned.csv'
#we use internal label for now
def get_label(subject_list):
  label_dict = {}
  with open(label_file) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
      if row['src_subject_id'] in subject_list:
        label = int(row['ksads_p_internal'])
        if label == 0:
          label_dict[row['src_subject_id']] = 0 #control
        else:
          label_dict[row['src_subject_id']] = 1
  return label_dict
