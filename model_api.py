from flask import Flask, request, jsonify
from lib.sam import SamPredictor
from lib.internvideo import InternVideoPredictor
import torch

app = Flask(__name__)

active_model = InternVideoPredictor()
zero_shot_model = SamPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    images = data['images']

    control = 0
    mode = "Zero"
    results = []

    for img in images:
        active_result = active_model.predict(img)
        if control == 0:
            zero_result = zero_shot_model.predict(img)
            if mode == "Active":
                results.append(active_result)
            else:
                results.append(zero_result)

            anno = zero_shot_model.predict(img)
            if mean_average_precision(active_result, anno) > mean_average_precision(zero_result, anno):
                control = 1
                mode = "Active"
            else:
                control = 0
                mode = "Zero"
        else:
            control -= 1
            results.append(active_result)

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
