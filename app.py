from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask, render_template, url_for, redirect, request, make_response
import numpy as np
from datetime import datetime
from functools import wraps, update_wrapper
import cv2
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]= '/static'


def colorize(image_path): 
    prototxt_path = "model/colorization_deploy_v2.prototxt"
    model_path = "model/colorization_release_v2.caffemodel"
    points_path = "model/pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts = np.load(points_path)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    #channels, extract the 'L' channel, and then
    # perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    # pass the L channel through the network which will *predict* the 'a'
    # and 'b' channel values
    print("[INFO] colorizing image...")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the
    # predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then
    # clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned
    # 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    # show the original and output colorized images
    cv2.imwrite('static/colorized.jpg',colorized)
    return True

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


@app.route('/')
@nocache
def main():
    params={
            'style': "/static/images/index.jpg",
            'attrib': "display:none;",
            'text' :"GET STARTED",
        }
    return render_template('index.html',**params)

@app.route('/mix', methods=['GET','POST'])
@nocache
def mix():
    if request.method == 'POST':
        if request.files:
            content_image = request.files["stylebtn"]
            content_image.save('static/images/content.jpg')
            if colorize('static/images/content.jpg'):
                path = '/static/colorized.jpg'
                params={
            'style': "/static/images/content.jpg",
            'attrib': "display:inline-block;",
            'text' :"GET STARTED",
            'result': path
        }
                return render_template('index.html',**params)
            return None
    return None

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

if __name__ == "__main__":
    app.run(debug=True)
