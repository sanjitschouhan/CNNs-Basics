import cv2, os
import numpy as np
from PIL import Image

import sys

def read_images (path , sz= None ):
	c = 0
	X,y = [], []
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
					X. append (np. asarray (im , dtype =np. uint8 ))
					y. append (c)
				except IOError :
					print "I/O error ({0}) : {1} "
				except :
					print " Unexpected error :", sys . exc_info () [0]
				
			c = c+1
			print c
	return [X,y]
recognizer = cv2.createLBPHFaceRecognizer()
im,la = read_images ("c:/att_faces")
recognizer.train(im, np.array(la))
for i in range(1, 5):
	t=str(i)+".pgm"
	predict_image_pil = Image.open(t).convert('L')
	predict_image = np.array(predict_image_pil, 'uint8')
#faces = faceCascade.detectMultiScale(predict_image)
#for (x, y, w, h) in faces:
	nbr_predicted, conf = recognizer.predict(predict_image)
	nbr_actual = 4
	if nbr_actual == nbr_predicted:
		print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
	else:
        	print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
	cv2.imshow("Recognizing Face", predict_image)
	cv2.waitKey(1000)



