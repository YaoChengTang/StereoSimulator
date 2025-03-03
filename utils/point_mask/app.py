from flask import Flask, request, send_file, render_template, redirect, url_for, jsonify
import os
import sys
import argparse
from PIL import Image
from io import BytesIO
from flask import Flask, send_file
import cv2
import numpy as np
app = Flask(__name__)

# 全局变量存储数据
image_data = {
    "path": "",
    "orig_width": 0,
    "orig_height": 0,
    "points": []
}

@app.route("/")
def index():
    return render_template("index.html")


from flask import send_file
import os

@app.route("/get_image")
def get_image():
    # 获取前端传来的图片路径参数
    global image_path
    image_path = request.args.get("path")
    return send_file(image_path)

@app.route("/load_image", methods=["POST"])
def load_image():
    global image_data, image_path
    image_path = request.json["image_path"]
    
    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "图片路径不存在"})
    
    # 获取原始图片尺寸
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(h,w)
    image_data = {
        "path": image_path,
        "orig_width": w,
        "orig_height": h,
        "points": []
    }
    
    return jsonify({
        "status": "success",
        "orig_width": w,
        "orig_height": h
    })

@app.route("/record_point", methods=["POST"])
def record_point():
    global image_data
    x = int(request.json["x"])
    y = int(request.json["y"])
    print(x,y)
    image_data["points"].append((x, y))
    return jsonify({"status": "success", "points": image_data["points"]})

@app.route("/save_mask", methods=["POST"])
def save_mask():
    global image_data, image_path
    save_name = request.json["save_name"]
    
    if not image_data["path"]:
        return jsonify({"status": "error", "message": "请先加载图片"})
    
    if not image_data["points"]:
        return jsonify({"status": "error", "message": "没有记录任何点"})

    # 创建原始尺寸的mask
    mask = np.zeros((image_data["orig_height"], image_data["orig_width"]), dtype=np.uint8)
    
        # 创建原始尺寸的mask
    points = np.array(image_data["points"], dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    # 使用 cv2.fillPoly 填充多边形区域为白色（255）
    cv2.fillPoly(mask, [points], color=255)
    print(os.path.dirname(image_path.replace("Dataset_new_structure","SAM")))
    os.makedirs(os.path.dirname(image_path.replace("Dataset_new_structure","SAM")), exist_ok=True)
    mask_path = f"{image_path.replace('Dataset_new_structure','SAM').replace('.png','')}-{save_name}-illusion.jpg"
    cv2.imwrite(mask_path, mask)
    image_data["points"] = []
    return jsonify({"status": "success", "mask_path": os.path.abspath(mask_path)})


@app.route("/save_mask1", methods=["POST"])
def save_mask1():
    global image_data, image_path
    save_name = request.json["save_name"]
    
    if not image_data["path"]:
        return jsonify({"status": "error", "message": "请先加载图片"})
    
    if not image_data["points"]:
        return jsonify({"status": "error", "message": "没有记录任何点"})

    # 创建原始尺寸的mask
    mask = np.zeros((image_data["orig_height"], image_data["orig_width"]), dtype=np.uint8)
    
        # 创建原始尺寸的mask
    points = np.array(image_data["points"], dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    # 使用 cv2.fillPoly 填充多边形区域为白色（255）
    cv2.fillPoly(mask, [points], color=255)
    os.makedirs(os.path.dirname(image_path.replace("Dataset_new_structure","SAM")), exist_ok=True)
    mask_path = f"{image_path.replace('Dataset_new_structure','SAM').replace('.png','')}-{save_name}-nonillusion.jpg"
    cv2.imwrite(mask_path, mask)
    image_data["points"] = []
    return jsonify({"status": "success", "mask_path": os.path.abspath(mask_path)})

if __name__ == "__main__":
   # 默认路径为空，用户可以通过输入框设置
    IMAGE_FOLDER = None

    # if len(sys.argv) == 2:
    #     IMAGE_FOLDER = sys.argv[1]

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="A script that accepts folder and port as input arguments")
    parser.add_argument(
        '-f', '--folder', 
        type=str, 
        default="./",
        help='Path to the folder'
    )
    parser.add_argument(
        '-p', '--port', 
        type=int, 
        required=True,  # port 是必需的
        help='Port number for the application'
    )
    args = parser.parse_args()
    
    IMAGE_FOLDER = args.folder
    PORT = args.port
    print(f"Folder path: {IMAGE_FOLDER}")
    print(f"Port number: {PORT}")

    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=PORT, debug=True)
