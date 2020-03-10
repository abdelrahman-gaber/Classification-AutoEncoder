import numpy as np
import os
import argparse

from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from Utils import CalculateConfusionMatrix, GetCifar10Mean, GetCifar10STD

parser = argparse.ArgumentParser(description='classification for cifar10 datset, and evaluate the trained model')
parser.add_argument('--model_path', help='full path to trained keras model', default='models/cls_cifar10/cls_cifar10.h5')
parser.add_argument('--data_dir', help='full path to test images, expects a folder with sub-folder for each class',
		 default='data/cifar10_keras/test/')
args = parser.parse_args()

data_dir = args.data_dir
model_path = args.model_path
cm_img_path = os.path.splitext(model_path)[0] + "_cm.jpg"

# cifar10 classes 
id_cls = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
target_names = ["airplane", "automobile", "bird","cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = load_model(model_path)

pred_list = []
true_name_list = []
pred_name_list = []
for class_dir in os.listdir(data_dir):
	class_path = os.path.join(data_dir, class_dir)
	if os.path.isdir(class_path):
		class_name = str(class_dir)
		for files in os.scandir(class_path):
			if files.is_file() and (files.name.endswith('.png') or files.name.endswith('.jpg')):

				image_pth = os.path.join(class_path, files.name)
				test_image_in= image.load_img(image_pth)
				test_image = image.img_to_array(test_image_in)
				test_image = np.expand_dims(test_image, axis = 0) * 1./255
				test_image = (test_image - GetCifar10Mean()) / GetCifar10STD()
				#print(test_image.shape)

				preds = model.predict(test_image)
				id_pred = np.argmax(preds)
				if id_cls[id_pred] == class_name:
					pred_list.append(1)
				else:
					pred_list.append(0)

				true_name_list.append(class_name)
				pred_name_list.append(id_cls[id_pred])
				#print(id_cls[id_pred] + " : " + str(preds[0][id_pred]) + " ... gt: " + class_name )

accuracy = pred_list.count(1) / len(pred_list)
print("Accuracy of the model '%s' with cifar10 test set is: "%(model_path) + str(accuracy) )

CalculateConfusionMatrix(true_name_list, pred_name_list, target_names, cm_img_path, accuracy)
