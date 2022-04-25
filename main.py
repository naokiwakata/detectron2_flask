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

app = Flask(__name__)

# Initialize
# データセットを登録
register_coco_instances(
    "leaf", {}, "PumpkinLeaf\PumpkinLeaf.json", "PumpkinLeaf/")
coins_metadata = MetadataCatalog.get("leaf")
setup_logger()

# 設定を決める
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation\\mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1クラスのみ

cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "C:\\Users\\wakanao\\detectron2_flask\\model_final.pth")  # 絶対パスでなければならないっぽい
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.DEVICE = "cpu"

# 予測器を作成
predictor = DefaultPredictor(cfg)
img_dir = "static/imgs/"


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

        outputs = predictor(img)

        dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        img_path = img_dir + dt_now + ".jpg"
        v = Visualizer(img[:, :, ::-1],
                       metadata=coins_metadata,
                       scale=1.0
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = v.get_image()[:, :, ::-1]
        cv2.imwrite(img_path, img)

    # 保存した画像ファイルのpathをHTMLに渡す
    return render_template('index.html', img_path=img_path)


if __name__ == '__main__':
    app.run()
