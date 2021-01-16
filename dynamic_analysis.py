# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

import numpy as np
import argparse
import matplotlib.pyplot as plt
#import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

all_data_list = list()
# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}



# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
def analyze(video_name,candidateUniqueId):
	#vs = VideoStream(src=video_name).start()
	#vs = VideoStream(src="http://huntiillvideo.s3.ap-south-1.amazonaws.com/test.webm").start()
	vs = cv2.VideoCapture(video_name)

	#time.sleep(2.0)
	#fourcc = cv2.VideoWriter_fourcc(*'XVID') 
	#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		ret,frame = vs.read()
		if ret != True:
			break
		try:
			frame = imutils.resize(frame, width=400)
		except:
			break

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence < 0.9:
				continue

			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
 			
			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX-20, startY-20), (endX+20, endY+20),(0, 0, 255), 2)
			#cv2.imwrite('./down/'+str(time.time())+'.jpg',frame[startY-20:endY+20,startX-20:endX+20])
			crop_frame = frame[startY-20:endY+20,startX-20:endX+20]
			emoji = ''
			try:
				gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
				cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
				prediction = model.predict(cropped_img)
				maxindex = int(np.argmax(prediction))
				print(emotion_dict[maxindex])
				emoji = emotion_dict[maxindex]
				all_data_list.append(emoji)
			except:
				print('err')
			#cv2.putText(frame, text+' '+emoji, (startX, y),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		# show the output frame
		#cv2.imshow("Frame", cv2.resize(frame,(1200,700)))
	#out.write(cv2.resize(frame,(640, 480)))

	print(len(all_data_list))
	total_data_points = len(all_data_list)
	Angry_count = 1
	Fearful_count = 1
	Disgusted_count = 1
	Happy_count = 1
	Neutral_count = 1
	Sad_count = 1
	Surprised_count = 1

	#{0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

	for data in all_data_list:
		if data == 'Angry':
			Angry_count = Angry_count + 1
		elif data == 'Disgusted':
			Disgusted_count = Disgusted_count + 1
		elif data == 'Fearful':
			Fearful_count = Fearful_count + 1
		elif data == 'Happy':
			Happy_count = Happy_count + 1
		elif data == 'Neutral':
			Neutral_count = Neutral_count + 1
		elif data == 'Sad':
			Sad_count = Sad_count + 1
		elif data == 'Surprised':
			Surprised_count = Surprised_count + 1
		else:
			print('skiped.......')

	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_axes([0,0,1,1])
	langs = ["Angry-"+str(round((Angry_count/total_data_points)*100))+"%", "Disgusted-"+str(round((Disgusted_count/total_data_points)*100))+"%", "Fearful-"+str(round((Fearful_count/total_data_points)*100))+"%", "Happy-"+str(round((Happy_count/total_data_points)*100))+"%", "Neutral-"+str(round((Neutral_count/total_data_points)*100))+"%", "Sad-"+str(round((Sad_count/total_data_points)*100))+"%", "Surprised-"+str(round((Surprised_count/total_data_points)*100))+"%"]

	counts = [Angry_count, Disgusted_count, Fearful_count, Happy_count, Neutral_count, Sad_count, Surprised_count]
	support_Angry = 1
	try:
		support_Angry =round((Angry_count/total_data_points)*100)
	except Exception as e:
		print(e)
	
	support_Disgusted = 1
	try:
		support_Disgusted = round((Disgusted_count/total_data_points)*100)
	except Exception as e:
		print(e)
	
	support_Fearful = 1
	try:	
		support_Fearful = round((Fearful_count/total_data_points)*100)
	except Exception as e:
		print(e)

	support_Happy = 1
	try:
		support_Happy = round((Happy_count/total_data_points)*100)
	except Exception as e:
		print(e)
	
	support_Neutral = 1
	try:
		support_Neutral = round((Neutral_count/total_data_points)*100)
	except Exception as e:
		print(e)
	
	support_Sad = 1
	try:
		support_Sad = round((Sad_count/total_data_points)*100)
	except Exception as e:
		print(e)
	
	support_Surprised = 1
	try:
		support_Surprised = round((Surprised_count/total_data_points)*100) 
	except Exception as e:
		print(e)	

	conf_Angry = 1
	try:
		conf_Angry =round((total_data_points/Angry_count)*100)
	except Exception as e:
		print(e)

	conf_Disgusted = 1
	try:
		conf_Disgusted = round((total_data_points/Disgusted_count)*100)
	except Exception as e:
		print(e)

	conf_Fearful = 1
	try:
		conf_Fearful = round((total_data_points/Fearful_count)*100)
	except Exception as e:
		print(e)

	conf_Happy = 1
	try:
		conf_Happy = round((total_data_points/Happy_count)*100)
	except Exception as e:
		print(e)

	conf_Neutral = 1
	try:
		conf_Neutral = round((total_data_points/Neutral_count)*100)
	except Exception as e:
		print(e)

	conf_Sad = 1
	try:
		conf_Sad = round((total_data_points/Sad_count)*100)
	except Exception as e:
		print(e)

	conf_Surprised = 1
	try:
		conf_Surprised = round((total_data_points/Surprised_count)*100)

	except Exception as e:
		print(e)

	lift_Angry = 1
	try:
		lift_Angry = conf_Angry//support_Angry
	except Exception as e:
		print(e)

	lift_Disgusted = 1
	try:
		lift_Disgusted = conf_Disgusted//support_Disgusted
	except Exception as e:
		print(e)

	lift_Fearful = 1
	try:
		lift_Fearful = conf_Fearful//support_Fearful
	except Exception as e:
		print(e)
	
	lift_Happy = 1
	try:
		lift_Happy = conf_Happy//support_Happy
	except Exception as e:
		print(e)
	
	lift_Neutral = 1
	try:
		lift_Neutral = conf_Neutral//support_Neutral
	except Exception as e:
		print(e)
	
	lift_Sad = 1
	try:
		lift_Sad = conf_Sad//support_Sad
	except Exception as e:
		print(e)

	lift_Surprised  = 1
	try:
		lift_Surprised = conf_Surprised//support_Surprised
	except Exception as e:
		print(e)
	print(lift_Angry,lift_Disgusted,lift_Fearful,lift_Happy,lift_Neutral,lift_Sad,lift_Surprised)
	names = ['lift_Angry','lift_Disgusted','lift_Fearful','lift_Happy','lift_Neutral,lift_Sad','lift_Surprised']
	#ax.bar(names,[lift_Angry,lift_Disgusted,lift_Fearful,lift_Happy,lift_Neutral,lift_Sad,lift_Surprised])
	fig = plt.figure(figsize =(10, 7)) 
	plt.pie(counts, labels = langs) 
	plt.xlabel('lift_Angry = '+str(lift_Angry)+', lift_Disgusted ='+str(lift_Disgusted)+', lift_Fearful='+str(lift_Fearful)+', lift_Happy='+str(lift_Happy)+', lift_Neutral='+str(lift_Neutral)+', lift_Sad='+str(lift_Sad)+', lift_Surprised='+str(lift_Surprised)) 
	#plt.pie([lift_Angry,lift_Disgusted,lift_Fearful,lift_Happy,lift_Neutral,lift_Sad,lift_Surprised], labels = names) 
	#plt.savefig(candidateUniqueId+'.png')

	# do a bit of cleanup
	#cv2.destroyAllWindows()
	#vs.stop()
	vs.release()
	#aaa
	#return video_name.split('.')[0]+'.png'
	lift = {'liftAngry':lift_Angry, 'liftDisgusted':lift_Disgusted, 'liftFearful':lift_Fearful, 'liftHappy':lift_Happy, 'liftNeutral':lift_Neutral, 'liftSad':lift_Sad, 'liftSurprised' :lift_Surprised}
	support = {"angry":(round((Angry_count/total_data_points)*100)), "disgusted":(round((Disgusted_count/total_data_points)*100)), "fearful":(round((Fearful_count/total_data_points)*100)), "happy":(round((Happy_count/total_data_points)*100)), "neutral":(round((Neutral_count/total_data_points)*100)), "sad":(round((Sad_count/total_data_points)*100)), "surprised":(round((Surprised_count/total_data_points)*100))}
	print({'support':support,'lift':lift})
	return {'support':support,'lift':lift}
