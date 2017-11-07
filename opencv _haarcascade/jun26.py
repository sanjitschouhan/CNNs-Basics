import cv2, os
import numpy as np
from PIL import Image

import sys
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

def read_images (path , sz= None ):
	c = 0
	X,y1 = [], []
	for dirname , dirnames , filenames in os. walk ( path ):
		for subdirname in dirnames :
			subject_path = os. path . join ( dirname , subdirname )
			for filename in os. listdir ( subject_path ):
				try :
					im = Image . open (os. path . join ( subject_path , filename ))
					im = im. convert ("L")
# resize to given size (if given )
					if (sz is not None ):
						im = im. resize (sz , Image . ANTIALIAS )
					tt=np. asarray (im , dtype =np. uint8 )

					faces = faceCascade.detectMultiScale(tt)
        # If face is detected, append the face to images and the label to labels
        				for (x, y, w, h) in faces:
            					X.append(tt[y: y + h, x: x + w])
						y1. append (c)
            
#labels.append(nbr)
            					cv2.imshow("Adding faces to traning set...", tt[y: y + h, x: x + w])
            					cv2.waitKey(50)

					#X. append (tt)
						
				except IOError :
					print "I/O error ({0}) : {1} "
				except :
					print " Unexpected error :", sys . exc_info () [0]
				
			c = c+1
			print c
	return [X,y1]
recognizer = cv2.createLBPHFaceRecognizer()
im,la = read_images ("c:/att_faces")
recognizer.train(im, np.array(la))
for i in range(1, 5):
	t=str(i)+".gif"
	predict_image_pil = Image.open(t).convert('L')
	predict_image = np.array(predict_image_pil, 'uint8')
	faces = faceCascade.detectMultiScale(predict_image)
	for (x, y, w, h) in faces:
		nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
	nbr_actual = 4
	if nbr_actual == nbr_predicted:
		print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
	else:
        	print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
	cv2.imshow("Recognizing Face", predict_image)
	cv2.waitKey(1000)




