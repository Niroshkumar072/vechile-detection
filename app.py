import cv2 

#image and video file 
img = cv2.imread('traffic_jam.jpg')
video = cv2.VideoCapture('carchase.mp4')

#trained classifier
classifier = 'cars.xml'

#cassififer
tracker = cv2.CascadeClassifier(classifier)

#bw image bw(black and white)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detector
cars = tracker.detectMultiScale(black_n_white)


while True:

    #reads frame by frame
    (sucess, frame) = video.read()

    if sucess:
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detector
    cars = tracker.detectMultiScale(gray_scale)

    #Draw Rectangle for cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y),(x+w , y+h), (0,255,0),2)

    #Display img
    cv2.imshow('Vechile Detector', frame) #change to video whenever you want by changing img to video inside the bracket

    #key press to close
    key = cv2.waitKey(1)



    #Exit
    if key==81 or key==113:
        break

# print(cars)

# #Draw Rectangle for cars
# for (x,y,w,h) in cars:
#     cv2.rectangle(img, (x,y),(x+w , y+h), (0,255,0),2)

# #Display img
# cv2.imshow('Vechile Detector', video) #change to video whenever you want by changing img to video inside the bracket

# #key press to close
# cv2.waitKey(1)

# print("done")
