# image_compression.py compresses raw images for the following registration process
# Please see the help.txt file for details.
#
# Copyright (C) 2017, Tatsuya C. Murakami
#    
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
import sys
from tifffile import imsave
from time import localtime, strftime
import glob
import numpy as np
import math
import os
import csv

def image_compression(raw_folder,sub_folder_name,imtype,xy_compression_ratio,z_compression_ratio,output_path):
    image_folder = raw_folder + "/" + sub_folder_name + "/"
    file_list = glob.glob(image_folder + "*." + imtype)
    file_list.sort()
    y, x = cv2.imread(file_list[0], readmode).shape
    z = len(file_list)

    comp_x = int(round(x/xy_compression_ratio,0))
    comp_y = int(round(y/xy_compression_ratio,0))
    comp_z = int(math.floor(z/z_compression_ratio))
    size = (comp_x, comp_y)

    k = 0
    compress_tif = np.zeros((comp_z,comp_y,comp_x), 'uint16')
    for file in file_list:
        img = cv2.imread(file_list[k*int(z_compression_ratio)], readmode)
        compress_tif[k,:,:] = cv2.resize(img,size)
        k += 1
        if k >= comp_z or k*int(z_compression_ratio) >= z:
            break
    imsave(output_path + "/" + sub_folder_name + ".tif",compress_tif)
    return [x,y,z,comp_x,comp_y,comp_z]


if len(sys.argv)!=2:
    print "\nUsage: "+sys.argv[0]+" <parameter_file>"
    quit()

# Reading the parameters
parameter_file = open(sys.argv[1],'r')
parameters = []
for line in parameter_file:
    if line[0] == "#":
        continue
    parameters.append(line.rstrip())
parameter_file.close()

# Processing the parameters
raw_folder = parameters[0]
output_path = parameters[1]
xy_compression_ratio = float(parameters[2])
z_compression_ratio = float(parameters[3])

# Checking the parameters
print "The method will: "
print " - read nucleus stained images from "+raw_folder
print " - save resulting data to "+output_path
print " - compress images in xy direction: "+str(xy_compression_ratio)
print " - compress images in z direction: "+str(z_compression_ratio)
print " "
while 1:
    feedback = raw_input("Is this correct? (yes/no)\t").rstrip()
    if feedback == "yes":
        print "Program starting...\n"
        break
    if feedback == "no":
        print "Please edit the parameter file."
        quit()

# Make output directory
if not(os.path.exists(output_path)):
    os.mkdir(output_path)

# Fixed parameters
imtype = "tif"
readmode = -1
log = output_path + "/log_step1.txt"
log_file = open(log,'w')
compression_info = {}
sub_folder_list = os.listdir(raw_folder + "/")
sub_folder_list.sort()

# Start compression
for sub_folder in sub_folder_list:
    output_message = strftime("%H:%M:%S", localtime())+": Image compression of "+sub_folder 
    print output_message
    log_file.write(output_message+"\n")
    size_info = image_compression(raw_folder,sub_folder,imtype,xy_compression_ratio,z_compression_ratio,output_path)
    compression_info[sub_folder] = size_info

# Write image size information
with open(output_path + "/size_info.csv", "a") as f_write:
    writer = csv.writer(f_write, lineterminator='\n')
    for key in compression_info:
        write_list = compression_info[key]
        write_list.insert(0,key)
        writer.writerow(write_list)
    output_message = strftime("%H:%M:%S", localtime())+": Done." 
    print output_message
    log_file.write(output_message+"\n")
