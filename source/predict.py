# -*- coding: utf-8 -*-
#   ___      _ _                    _     
#  / _ \    | | |                  | |    
# / /_\ \ __| | |__   ___  ___  ___| |__  
# |  _  |/ _` | '_ \ / _ \/ _ \/ __| '_ \ 
# | | | | (_| | | | |  __/  __/\__ \ | | |
# \_| |_/\__,_|_| |_|\___|\___||___/_| |_|
# Date:   2021-03-18 00:14:04
# Last Modified time: 2021-03-18 02:05:16

from load_json import get_class_id
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2

class Predict():
	def __init__(self, image):
		self.image = image
		self.output = image
		self.preprocess_image()
		self.id,self.label,self.prob = self.classify_image()

	def preprocess_image(self):
		self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		self.image = preprocess_input(self.image)
		self.image = cv2.resize(self.image, (224,224))
		self.image = np.expand_dims(self.image, axis=0)

	def classify_image(self):
		print("Predicting image...")
		self.model = ResNet50(weights="imagenet")
		preds = self.model.predict(self.image)
		preds = decode_predictions(preds, top = 3)[0]

		for (i, (id, label, prob)) in enumerate(preds):
			if i == 0:
				print(f"[SELECTED] {label} => {get_class_id(label)}")
			print(f"[TOP 3] {i+1}. {label}: {prob}%")
		print("\n")

		return get_class_id(preds[0][1]), preds[0][1], preds[0][2]

	def display_image(self,i):
		self.prob = str(self.prob*100)
		self.prob = self.prob[:6]

		text = f"{self.label}: {self.prob}%"
		cv2.putText(self.output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
		cv2.imshow(f"image_{i}", self.output)
		
		cv2.imwrite(f"../Images/image_{i}.jpg", self.output)

if __name__ == '__main__':
	image_filename="../dataset/pig.jpg"
	# image_filename="Dataset/adversarial.png"
	image=cv2.imread(image_filename)

	p=Predict(image)
	p.display_image()