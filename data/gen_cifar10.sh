# 1. Download cifar10 (source: https://pjreddie.com/projects/cifar-10-dataset-mirror/)
wget http://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz
rm cifar.tgz

# 2. Add the images to folders with class names, which is neede for keras "flow_from_directory" function
# train split
python process_cifar10.py --images_dir cifar/train/ --out_dir cifar10_keras/train/ 
# test split
python process_cifar10.py --images_dir cifar/test/ --out_dir cifar10_keras/test/
