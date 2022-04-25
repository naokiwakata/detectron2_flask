import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from flask import Flask, render_template, request
import numpy as np
import datetime

from predictor import Predictor

app = Flask(__name__)

predictor = Predictor()


@app.route('/', methods=["GET", "POST"])
def predict_img():
    if request.method == 'GET':
        img_path = None
    elif request.method == 'POST':

        # POSTにより受け取った画像を読み込む
        stream = request.files['file'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        # 現在時刻を名前として「imgs/」に保存する

        img = np.clip(img, 0, 255).astype(np.uint8)

        predictor.predict(img=img)
        img = predictor.img

        dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        _img_dir = "static/imgs/"
        img_path = _img_dir + dt_now + ".jpg"
        cv2.imwrite(img_path, img)

    # 保存した画像ファイルのpathをHTMLに渡す
    return render_template('index.html', img_path=img_path)


if __name__ == '__main__':
    app.run()
