from flask import Flask, request, jsonify, send_from_directory
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/detect', methods=['POST'])
def detect_object():
    data = request.get_json()
    image = data.get("image", None)
    if image:
        # Process the image (placeholder function)
        return jsonify({"result": "Example object"})
    return jsonify({"result": "No image provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
