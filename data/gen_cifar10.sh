# 1. Download cifar10 (source: https://pjreddie.com/projects/cifar-10-dataset-mirror/)
wget http://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz
rm cifar.tgz

# 2. Add the images to folders with class names, which is neede for keras 
# In addition, split the dataset to have specific number of samples for some classes given in text file
# train split
python process_cifar10.py --images_dir cifar/train/ --out_dir cifar10_keras/train/ --number_classes_given --label_number_list cifar10_train_label_number.txt
# test split
python process_cifar10.py --images_dir cifar/test/ --out_dir cifar10_keras/test/


# 3. move random 100 images from each class in train folder and use them as our validation split
dataset_path=cifar10_keras/train/
save_path=cifar10_keras/val
# Find all the leaf folders in the dataset path and stores them in an array
data_folders=( $(find $dataset_path -type d -mindepth 1 -links 2) )
# Now we iterate through all the image folders
for folder in "${data_folders[@]}"
do
    source_folder=$folder
    save_folder=$save_path/${folder#${dataset_path}}
    # Create the folder if one does not exist already
    mkdir -p $save_folder
	x=$(find $source_folder -type f | shuf -n 100)
	for file in $x
	do
    	mv $file $save_folder
	done 
done

echo "Data generation done"