import os
import argparse
import shutil # to copy files

def ParseLabelNumbersFile(filename):
	'''
		This function read the text file that contains number of images for each class, 
		and return a dictionary with "class name : number of images"
	'''
	label_number_dict = dict()
	f = open(filename)
	lines = f.read().splitlines()
	for l in lines:
		values = l.split(" ")
		class_label =  values[0]
		max_numbers = int(values[1])
		label_number_dict[class_label] = max_numbers
	f.close()

	return label_number_dict


parser = argparse.ArgumentParser(description='Processing cifar10 dataset.')
parser.add_argument('--images_dir', default='cifar/test/')
parser.add_argument('--out_dir', default='cifar10_keras/test/')
parser.add_argument('--number_classes_given', dest='list_given', action='store_true', 
						help='This argument allows to choose if you want all the images from the dataset (default) \
                    	or predefined number of images for some classes. You need to specify the list if you enabled this argument.')
parser.add_argument('--label_number_list', default='cifar10_train_label_number.txt', help='Path to the txt file contating labels \
						and the desired number of samples for each class label.')
parser.set_defaults(list_given=False)
args = parser.parse_args()

if args.list_given:
	if os.path.isfile(args.label_number_list):
		label_number_dict = ParseLabelNumbersFile(args.label_number_list)
		print(label_number_dict)
	else:
		print("A text file with max number of images for each class must be provided if you enable the argument --number_classes_given. \
		 \nPlease provide the crrect path or disable the argument --number_classes_given ")
		quit()

if not os.path.exists(args.out_dir):
			os.makedirs(args.out_dir)

for file in os.scandir(args.images_dir):
	if file.is_file() and file.name.endswith('.png'):
		# file name is in format: "number_classname.png"
		file_name = os.path.splitext(file.name)[0] # remove the extension
		class_name = file_name.split("_")[-1] # parse class name  
		output_dir = os.path.join(args.out_dir, class_name)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		# copy file to its class folder in the given output directory
		if args.list_given:
			if label_number_dict[class_name] > 0:
				shutil.copy(os.path.join(args.images_dir, file.name), output_dir)
				label_number_dict[class_name] = label_number_dict[class_name] - 1 # decrease 1 from the max number
		else:
			shutil.copy(os.path.join(args.images_dir, file.name), output_dir)