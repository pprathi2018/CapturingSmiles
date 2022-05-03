import numpy as np
import cv2 as cv
from tensorflow import keras
import tensorflow as tf
import imutils
from PIL import Image
import os

def convert_prediction(prediction):
  if prediction > 0.5:
    return 1
  else:
    return 0 

def main():
  cap = cv.VideoCapture(0)

  model = keras.models.load_model('AlexNet_CNN_model')
  cascade_face = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

  counter = 0
  smile_counter = 0

  if not os.path.exists('./video_capture/'):
    os.mkdir('./video_capture')

  if not cap.isOpened():
      print("Cannot open camera")
      os.rmdir('./video_capture')
      exit()
  while True:
      # Capture frame-by-frame
      ret, frame = cap.read()
      # if frame is read correctly ret is True
      if not ret:
          print("Can't receive frame (stream end?). Exiting ...")
          break
      SCALE_FACTOR = 1.1
      MIN_NEIGHBORS = 5
      frame = imutils.resize(frame, width=450)
      grey_frame = Image.fromarray(frame).convert('LA')
      cv2_image = cascade_face.detectMultiScale(frame, SCALE_FACTOR, MIN_NEIGHBORS)
      if len(cv2_image) != 0:

        x = cv2_image[0][0]
        y = cv2_image[0][1]
        w = cv2_image[0][2]
        h = cv2_image[0][3]
        area = (x, y, x+w, y+h)
        cropped = grey_frame.crop(area)
        saved_img_path = f'./video_capture/cropped_{counter}_ex.png'
        cropped.save(saved_img_path)

        curr_image = cv.imread(saved_img_path).astype('float32')
        resized = np.array([cv.resize(curr_image, (227, 227))])
        prediction = convert_prediction(model.predict(resized)[0][0])
        if prediction == 1:
          smile_counter += 1
          if smile_counter >= 10:
            ret, frame = cap.read()
            frame2 = frame.copy()
            img_name = 'captured_smile.png'
            cv.imwrite(f'./results/{img_name}', frame)
            os.remove(f'./video_capture/cropped_{counter}_ex.png')
            break
          else:
            os.remove(f'./video_capture/cropped_{counter}_ex.png')
        else:
          os.remove(f'./video_capture/cropped_{counter}_ex.png')
          smile_counter = 0
        counter += 1

      # Display the resulting frame
      cv.imshow('frame', frame)
      if cv.waitKey(1) == ord('q'):
          break
  # When everything done, release the capture
  cap.release()
  cv.destroyAllWindows()
  os.rmdir('./video_capture')

if __name__ == '__main__':
  main()
