# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
from demo import *

import os

import numpy as np

from flask import Flask
from flask import request
from flask import render_template




app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\GOKUL\Desktop\Deploying\deep-learning-master\static"
DEVICE = "cpu"
MODEL = None



@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            #pred = predict(image_location, MODEL)[0]
            img = detect_cv2('cfg/yolo-obj.cfg', 'model.weights', image_location)
            pred = "Detection Results"
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    #arch = EfficientNet.from_pretrained('efficientnet-b1')
    #model = Net(arch=arch, n_meta_features=len(meta_features))
    #MODEL.load_state_dict(torch.load("model.pth"))
    #app.run(host="0.0.0.0", port=12000, debug=True)
    app.run(debug=True)
