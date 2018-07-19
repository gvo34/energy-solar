import os
import io
import numpy as np

# import keras
# from keras.preprocessing import image
# from keras.preprocessing.image import img_to_array
# from keras.applications.xception import (
#     Xception, preprocess_input, decode_predictions)
# from keras import backend as K

from flask import Flask, request, redirect, url_for, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Data'

model = None
graph = None


# def load_model():
#     global model
#     global graph
#     model = Xception(weights="imagenet")
#     graph = K.get_session().graph


#load_model()


# def prepare_image(img):
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     # return the processed image
#     return img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        
        # print("THIS IS REQUEST")
        # print(request.form)

        # if request.files.get('file'):
        #     # read the file
        #     file = request.files['file']
        #     print(file)
        #     # read the filename
        #     filename = file.filename
        #     print(filename)
        #     # create a path to the uploads folder
        #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     file.save(filepath)

        # print("process file ", filepath)

        # Load the saved image using Keras and resize it to the Xception
        #format of 299x299 pixels
        # image_size = (299, 299)
        # im = keras.preprocessing.image.load_img(filepath,
        #                                         target_size=image_size,
        #                                         grayscale=False)
        # preprocess the image and prepare it for classification
        # image = prepare_image(im)
        # global graph
        # with graph.as_default():
        #     preds = model.predict(image)
        #     results = decode_predictions(preds)
        #     data["predictions"] = []

        #     # loop over the results and add them to the list of
        #     # returned predictions
        #     for (imagenetID, label, prob) in results[0]:
        #         r = {"label": label, "probability": float(prob)}
        #         data["predictions"].append(r)
        #     data["file"]=filepath
        #     # indicate that the request was a success
        #     data["success"] = True
        return render_template('index.html', predictions="")
    print("NOT A POST")
    return render_template('index.html',predictions="")

if __name__ == "__main__":
    app.run(debug=True)
