# cell_mapping.py gives annotation to the detected cells through registering
# the sample brain to the CUBIC-Atlas.
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

import os
import csv
from time import localtime, strftime
import numpy as np
import math
import json
import sys
from subprocess import call
import cv2

def readdist(xyzfilename_list,z_size,y_size,x_size):
	mv_img = np.zeros((3,z_size, y_size, x_size))
	xyz_counter = 0
	for filename in xyzfilename_list:
		f = open(filename, "r")
		flag = 0
		slice_num = 0
		for row in f:
			if flag == y_size:
				flag = 0
				slice_num += 1
				continue
			if slice_num == z_size:
				break

			row_list = row.split()
			row_list.reverse()

			mv_img[xyz_counter,slice_num,flag,:] = np.array(row_list) #mv_img[z, y, x] order
			flag += 1
		f.close()
		xyz_counter += 1
	return mv_img


def affinematrix_calc(filename):
	affine = np.zeros((4,4))
	f = open(filename, 'r')
	affine_file = f.read()
	f.close()

	lines = affine_file.split('\n')
	Parameter = lines[3].split(' ')
	FixedParameter = lines[4].split(' ')

	matrix = [[float(Parameter[1]),float(Parameter[2]),float(Parameter[3])],[float(Parameter[4]),float(Parameter[5]),float(Parameter[6])],[float(Parameter[7]),float(Parameter[8]),float(Parameter[9])]]
	translation = [float(Parameter[10]),float(Parameter[11]),float(Parameter[12])]
	center = [float(FixedParameter[1]),float(FixedParameter[2]),float(FixedParameter[3])]

	offset = [0, 0, 0]
	for i in range(0,3):
		offset[i] = translation[i] + center[i]
		for j in range(0,3):
			offset[i] -= matrix[i][j] * center[j]

	inv_matrix = np.array([[float(Parameter[1]),float(Parameter[2]),float(Parameter[3]),float(offset[0])],[float(Parameter[4]),float(Parameter[5]),float(Parameter[6]),float(offset[1])],[float(Parameter[7]),float(Parameter[8]),float(Parameter[9]),float(offset[2])],[0,0,0,1]])
	affine_matrix = np.linalg.inv(inv_matrix)

	affine = affine_matrix

	return affine

def ANTSregistration(iterationL,iterationM,iterationN,output_path,atlas_img,compress_img_path):
	cmd = "ANTS 3 -i " + str(iterationL) + "x" + str(iterationM) + "x" + str(iterationN) + \
	" -o " + output_path + "registered.nii --MI-option 64x300000 " + \
	"-m CC[" + atlas_img + "," + compress_img_path + ",1,5]" 
	call([cmd], shell=True)

	cmd = "WarpImageMultiTransform 3 " + compress_img_path + " " + output_path + "registered.tif -R " + atlas_img + " " + output_path + "registeredWarp.nii " + output_path + "registeredAffine.txt"
	call([cmd], shell=True)

	cmd = "fsl2ascii " + output_path + "registeredInverseWarp.nii " + output_path + "registeredInverseWarp.txt" 
	call([cmd], shell=True)

	affine_path = output_path + "registeredAffine.txt"
	moving_path = output_path + "registeredInverseWarp.txt"
	moving_list = [moving_path + "00000",moving_path + "00001",moving_path + "00002"]

	return [affine_path,moving_list]

def atlasmapping(output_path,csv_path,coordinate_info,affine,mv_img,x_size,y_size,z_size,compression_x,compression_y,compression_z):
	output_csv_name = output_path + "registered_points.csv"
	with open(output_csv_name,'a') as f_write:
		with open(csv_path, 'r') as f_read:
			reader = csv.reader(f_read)
			headers = reader.next()
			x_index = headers.index(coordinate_info[0])
			y_index = headers.index(coordinate_info[1])
			z_index = headers.index(coordinate_info[2])
			counter = 0
			for k in reader:
				counter += 1
				x = float(k[x_index]) / compression_x
				y = float(k[y_index]) / compression_y
				z = float(k[z_index]) / compression_z
				RX = x * affine[0][0] + y * affine[0][1] + z * affine[0][2] + affine[0][3]
				RY = x * affine[1][0] + y * affine[1][1] + z * affine[1][2] + affine[1][3]
				RZ = x * affine[2][0] + y * affine[2][1] + z * affine[2][2] + affine[2][3]
				X = int(RX)
				Y = int(RY)
				Z = int(RZ)
				if RX >= 0 and X + 1 < int(x_size) and RY >= 0 and Y + 1 < int(y_size) and RZ >= 0 and Z + 1 < int(z_size):
					#following seems complicated, but it calculates (linear interpolation of each point with mv_img)
					SyN_x = RX + (mv_img[0,Z,Y,X] * (1-(RX - X)) + mv_img[0,Z,Y,X+1] * (RX - X))
					SyN_y = RY + (mv_img[1,Z,Y,X] * (1-(RY - Y)) + mv_img[1,Z,Y+1,X] * (RY - Y))
					SyN_z = RZ + (mv_img[2,Z,Y,X] * (1-(RZ - Z)) + mv_img[2,Z+1,Y,X] * (RZ - Z))
					writer = csv.writer(f_write, lineterminator='\n')
					writer.writerow([round(SyN_x,5),round(SyN_y,5),round(SyN_z,5),counter])

def id_finder(indict,id_number):
	if isinstance(indict,dict):
		for key, value in indict.items():
			if isinstance(value, list):
				#print "yes"
				if value == []:
					pass
				else:
					for d in value:
						try:
							return id_finder(d,id_number)
						except ValueError:
							pass
			elif key == 'id':
				if value == id_number:
					return [indict['name'], indict['parent_structure_id']]
					#return indict['parent_structure_id']
	raise ValueError("Request file not found")

def color_finder(indict,id_number):
	if isinstance(indict,dict):
		for key, value in indict.items():
			if isinstance(value, list):
				if value == []:
					pass
				else:
					for d in value:
						try:
							return color_finder(d,id_number)
						except ValueError:
							pass
			elif key == 'id':
				if value == id_number:
					return indict['color_hex_triplet']
	raise ValueError("Request file not found")

def annotation(output_path,array_x,array_y,array_z,whole_list):
	id_dic = {}
	input_csv = output_path + "registered_points.csv"
	with open(input_csv, 'r') as f_read:
		reader = csv.reader(f_read)
		for k in reader:
			x = int(float(k[0]))
			y = int(float(k[1]))
			z = int(float(k[2]))
			if x >= 0 and x < array_x and y >= 0 and y < array_y and z >= 0 and z < array_z:
				number = (z - 1) * array_x * array_y + (y - 1) * array_x + x
				cell_list_array = whole_list[number]
				zero_confirmation = cell_list_array.size
				if zero_confirmation != 0:
					row_num, column_num = cell_list_array.shape
					my_pos = np.array([float(k[0]),float(k[1]),float(k[2])])
					dist_temp_array = np.subtract(cell_list_array[:,0:3],my_pos)
					dist_temp_array = np.square(dist_temp_array)
					dist_array = np.sum(dist_temp_array, axis=1)
					min_index = np.argmin(dist_array)
					atlas_id = int(cell_list_array[min_index,3])
					id_dic[int(k[3])] = atlas_id
				else:
					atlas_id = 0
					id_dic[int(k[3])] = atlas_id
			else:
				atlas_id = 0
				id_dic[int(k[3])] = atlas_id
	return id_dic

def mergecsv(output_path,coordinate_info,csv_path,id_dic):
	annotated_csv = output_path + "result.csv"
	with open(annotated_csv,'a') as f_write:
		with open(csv_path, 'r') as f_read:
			reader = csv.reader(f_read)
			headers = reader.next()
			x_index = headers.index(coordinate_info[0])
			y_index = headers.index(coordinate_info[1])
			z_index = headers.index(coordinate_info[2])			
			counter = 0
			for k in reader:
				counter += 1
				x = float(k[x_index])
				y = float(k[y_index])
				z = float(k[z_index])
				if counter in id_dic:
					allocated_atlas_id = id_dic[counter]
				else:
					allocated_atlas_id = 0
				writer = csv.writer(f_write, lineterminator='\n')
				writer.writerow([x,y,z,allocated_atlas_id])

def count_number(output_path):
	annotated_csv = output_path + "result.csv"
	output_csv = output_path + "regional_result.csv"

	count_array = np.genfromtxt(annotated_csv,delimiter = ",")
	ID_count_array = count_array[:,3]
	ids, counts = np.unique(ID_count_array,return_counts = True)
	ids = ids.tolist()
	counts = counts.tolist()
	counter_dict = dict(zip(ids,counts))

	with open(output_csv,'a') as f_write:
		for child_id in unique_list:
			if child_id == 0:
				row = [[],0.0,"None",counter_dict[child_id]]
			else:
				row = []
				parent_id_list = []
				[name, parent] = id_finder(structure['msg'][0],child_id)
				while parent != None:
					parent_id_list.append(int(parent))
					[parent_name, parent] = id_finder(structure['msg'][0],parent)
				row.append(parent_id_list[::-1])
				row.append(child_id)
				row.append(name)
				if child_id in ids:
					row.append(counter_dict[child_id])
				else:
					row.append(0)
			writer = csv.writer(f_write, lineterminator='\n')
			writer.writerow(row)

def image_out(output_path,compress_x,compress_y,compress_z):
	if not os.path.isdir(output_path + "annotated_points"):
		os.mkdir(output_path + "annotated_points")
	image_output_path = output_path + "annotated_points/"
	compress_x = int(compress_x)
	compress_y = int(compress_y)
	compress_z = int(compress_z)
	image_array = np.zeros((compress_z, compress_y, compress_x, 3),dtype = np.uint8)
	with open(output_path + "result.csv", 'r') as f_read:
		reader = csv.reader(f_read)
		for k in reader:
			z = int(float(k[2])/compression_z)
			y = int(float(k[1])/compression_y)
			x = int(float(k[0])/compression_x)
			atlas_id = int(float(k[3]))
			if z >= 0 and z < compress_z and y >= 0 and y < compress_y and x >= 0 and x < compress_x:
				if atlas_id == 0:
					BGR = "000000"
				else:
					BGR = str(color_finder(structure['msg'][0],int(float(k[3]))))
				image_array[z, y, x, 0] = int(BGR[0:2],16)
				image_array[z, y, x, 1] = int(BGR[2:4],16)
				image_array[z, y, x, 2] = int(BGR[4:6],16)

	for k in range(0, compress_z):
		str_k = str(k)
		while(len(str_k) < 5):
			str_k = "0" + str_k
		filename = image_output_path + "image" + str_k + ".tif"
		cv2.imwrite(filename, image_array[k,:,:,:])


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
output_path_parent = parameters[0]
atlas_folder = parameters[1]
size_info = parameters[2]
iterationL = parameters[3]
iterationM = parameters[4]
iterationN = parameters[5]
compress_img_path_parent = parameters[6]
csv_path_parent = parameters[7]

coordinate_info = [parameters[8],parameters[9],parameters[10]]
additional_info = []
for i in range(11,len(parameters)):
	additional_info.append(parameters[i])

# Checking the parameters
print "The method will: "
print " - save resulting data to "+output_path_parent
print " - read atlas from "+atlas_folder
print " - read size information from : " + size_info
print "\nANTS registration iteration: "
print iterationL,iterationM,iterationN
print " - read compressed nucleus stained image from "+compress_img_path_parent
print " - read csv from "+csv_path_parent
print " - read coordinate information from following column, X: "+coordinate_info[0]+" Y: "+coordinate_info[1]+" Z: "+coordinate_info[2]
print " - read additional information "+str(additional_info)########## test
print " "
while 1:
    feedback = raw_input("Is this correct? (yes/no)\t").rstrip()
    if feedback == "yes":
        print "Program starting...\n"
        break
    if feedback == "no":
        print "Please edit the parameter file."
        quit()


#Fixed inputs
atlas_folder = atlas_folder + "/"
x_size = 241.
y_size = 286.
z_size = 135.
atlas_img = atlas_folder + "reference120.tif"
atlas_csv = atlas_folder + "atlas120.csv"
allen_structure_path = atlas_folder + "structure_graph.json"
unique_id = atlas_folder + "uniqueID.csv"


#Make output folder
if not os.path.isdir(output_path_parent):
	os.mkdir(output_path_parent)

log = output_path_parent + "/log_step2.txt"
log_file = open(log,'w')

output_message = strftime("%H:%M:%S", localtime())+": Preparing for calculation "
print output_message
log_file.write(output_message+"\n")

#Obtain sample information
sample_info = []
with open(size_info, 'r') as f_read:
	reader = csv.reader(f_read)
	for k in reader:
		sample_info.append([k[0],float(k[1]),float(k[2]),float(k[3]),float(k[4]),float(k[5]),float(k[6])])

#Read structure graph 
f = open(allen_structure_path,'r')
structure = json.load(f)
f.close()

#Read unique ID in allen brain atlas annotation image
unique_array = np.genfromtxt(unique_id,delimiter = ",")
unique_list = unique_array.tolist()

#Read CUBIC-Atlas csv 
array_x = int(x_size)
array_y = int(y_size)
array_z = int(z_size)
whole_list = [np.empty((0,4))] * (array_x*array_y*array_z)
with open(atlas_csv, 'r') as f_read:
	reader = csv.reader(f_read)
	for k in reader:
		x = int(float(k[0]))
		y = int(float(k[1]))
		z = int(float(k[2]))
		if x >= 0 and x < array_x and y >= 0 and y < array_y and z >= 0 and z < array_z:
			number = (z - 1) * array_x * array_y + (y - 1) * array_x + x
			temp_array = np.append(whole_list[number],np.array([[float(k[0]),float(k[1]),float(k[2]),int(k[3])]]), axis = 0)
			whole_list[number] = temp_array

for sample in sample_info:
	sample_name = sample[0]

	output_message = sample_name
	print output_message
	log_file.write(output_message+"\n")
	
	compression_x = sample[1] / sample[4]
	compression_y = sample[2] / sample[5]
	compression_z = sample[3] / sample[6]
	compress_x = int(sample[4])
	compress_y = int(sample[5])
	compress_z = int(sample[6])
	output_path = output_path_parent + "/" + sample_name
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	output_path = output_path + "/"
	csv_path = csv_path_parent + "/" + sample_name + ".csv"
	compress_img_path = compress_img_path_parent + "/" + sample_name + ".tif"

	#ANTS symmmetric normalization registration
	output_message = strftime("%H:%M:%S", localtime())+": Registration to CUBIC-Atlas "
	print output_message
	log_file.write(output_message+"\n")

	[affine_path,moving_list] = ANTSregistration(iterationL,iterationM,iterationN,output_path,atlas_img,compress_img_path)

	#Application of deformation field to detected cells
	output_message = strftime("%H:%M:%S", localtime())+": Application of deformation field to detected cells "
	print output_message
	log_file.write(output_message+"\n") 

	affine = affinematrix_calc(affine_path)
	mv_img = readdist(moving_list,z_size,y_size,x_size)

	atlasmapping(output_path,csv_path,coordinate_info,affine,mv_img,x_size,y_size,z_size, compression_x, compression_y, compression_z)

	#Annotation to the detected cells
	output_message = strftime("%H:%M:%S", localtime())+": Annotation "
	print output_message
	log_file.write(output_message+"\n") 

	id_dictionary = annotation(output_path,array_x,array_y,array_z,whole_list)
	mergecsv(output_path,coordinate_info,csv_path,id_dictionary)

	# Count cell number in each annatomical region
	output_message = strftime("%H:%M:%S", localtime())+": Counting cell number in each anatomical region "
	print output_message
	log_file.write(output_message+"\n") 
	count_number(output_path)

	# Pritn image of annotated cells 
	output_message = strftime("%H:%M:%S", localtime())+": Image output "
	print output_message
	log_file.write(output_message+"\n")

	image_out(output_path,compress_x,compress_y,compress_z)


	output_message = strftime("%H:%M:%S", localtime())+": Done."
	print output_message
	log_file.write(output_message+"\n")


log_file.close()
