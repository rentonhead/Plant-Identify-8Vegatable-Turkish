import numpy as np
import cv2 as cv
from keras.preprocessing import image
import tensorflow as tf

frameCounter = 0
fps = 0

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

model = tf.keras.models.load_model("modelKeras/modelson.h5")

category = {0: 'Fasulye', 1: 'Brokoli', 2: 'Lahana', 3: 'Havuc', 4: "Karnabahar", 5: 'Salatalik', 6: 'Patates',
            7: 'Domates'}


def predict_image(image, model):

    img_processed = np.expand_dims(image, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    index = np.argmax(prediction)

    text = ("Sebze - {}".format(category[index]))

    return text

while True:

    ret, frame = cap.read()
    img = cv.resize(frame, (224, 224))
    test_image = image.img_to_array(img)
    text = predict_image(test_image, model)
    cv.putText(frame,text,(50,50),cv.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),2)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break
    if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv.destroyAllWindows()