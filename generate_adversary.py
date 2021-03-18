# -*- coding: utf-8 -*-
#   ___      _ _                    _     
#  / _ \    | | |                  | |    
# / /_\ \ __| | |__   ___  ___  ___| |__  
# |  _  |/ _` | '_ \ / _ \/ _ \/ __| '_ \ 
# | | | | (_| | | | |  __/  __/\__ \ | | |
# \_| |_/\__,_|_| |_|\___|\___||___/_| |_|
# Date:   2021-03-18 00:47:32
# Last Modified time: 2021-03-18 01:40:35

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
import numpy as np
import cv2

from predict import Predict

EPS = 2 / 255.0
LR = 0.1

def initialize(image):
	image = preprocess_image(image)
	model=ResNet50(weights='imagenet')
	optimizer=Adam(learning_rate=LR)
	scc_loss=sparse_categorical_crossentropy

	return image,model,optimizer,scc_loss

def preprocess_image(image):
	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	image=cv2.resize(image,(224,224))
	image=np.expand_dims(image,axis=0)
	return image

def clip_eps(tensor,eps):
	return tf.clip_by_value(tensor,eps,eps)

def create_noise(image):
	image=tf.constant(image,dtype=tf.float32)
	noise=tf.Variable(tf.zeros_like(image),trainable=True)
	return image,noise

def generate_adversary(model,image,noise,class_id,optimizer,scc_loss,steps=50):
	print("Generating Adversarial Image...")
	for step in range(0,steps):
		with tf.GradientTape() as tape:
			tape.watch(noise)

			adversary=preprocess_input(image+noise)
			preds=model(adversary,training=False)
			loss=-sparse_categorical_crossentropy(tf.convert_to_tensor([class_id]),preds)

			if step%5==0:
				print(f"step: {step}, loss: {loss.numpy()}")

		gradients=tape.gradient(loss,noise)
		optimizer.apply_gradients([(gradients,noise)])
		noise.assign_add(clip_eps(noise,EPS))
	print("\n")
	adver_image = (image + noise).numpy().squeeze()
	adver_image = np.clip(adver_image, 0, 255).astype("uint8")
	adver_image = cv2.cvtColor(adver_image, cv2.COLOR_RGB2BGR)

	return adver_image

if __name__=="__main__":
	image_filename="Dataset/pig.jpg"
	out_filename="Dataset/output.png"

	image=cv2.imread(image_filename)
	p=Predict(image)
	p.display_image(1)

	image,model,optimizer,scc_loss=initialize(image)
	class_id=int(p.id)
	image,noise=create_noise(image)
	adverary=generate_adversary(model,image,noise,class_id,optimizer,scc_loss)

	p=Predict(adverary)
	p.display_image(2)