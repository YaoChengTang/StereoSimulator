from flask import Flask, request, send_file, render_template, redirect, url_for, jsonify
import os
import sys
import argparse

app = Flask(__name__)

# 全局变量：指定的图片路径、图片列表、分页控制变量
IMAGE_FOLDER = None
images = []
current_page = 0  # 当前缩略图页面
start_idx = 0
images_per_page = 10  # 每页显示的图片数量
current_image_index = 0  # 当前查看的图片索引


@app.before_first_request
def load_images():
    """
    在第一次请求前加载初始文件夹（如果设置）
    """
    global images
    if IMAGE_FOLDER:
        load_images_from_folder(IMAGE_FOLDER)


def load_images_from_folder(folder_path):
    """
    动态加载指定文件夹的图片
    """
    global images
    valid_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    images = []
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Invalid folder path: {folder_path}")
        return

    # 加载图片文件并排序
    def extract_int(filename):
        """
        从文件名中提取整数部分用于排序
        """
        try:
            return int(os.path.splitext(filename)[0][7:])  # 提取文件名（去掉扩展名）并转换为整数
        except ValueError:
            return float('inf')  # 如果无法转换为整数，放在最后

    images = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    images.sort(key=extract_int)  # 按整数部分排序
    print(f"[INFO] Loaded {len(images)} images from {folder_path}")


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    显示缩略图页面并提供更改文件夹的功能
    """
    global start_idx, IMAGE_FOLDER

    # 如果是提交表单（更改文件夹路径）
    if request.method == 'POST':
        new_folder = request.form.get('image_folder')
        if new_folder:
            IMAGE_FOLDER = new_folder.strip()
            start_idx = 0  # 重置分页
            load_images_from_folder(IMAGE_FOLDER)


    # 计算当前页的图片范围
    start = start_idx
    end = start_idx + images_per_page
    thumbnails = images[start:end]

    # 判断是否有下一页
    has_next_page = end < len(images)

    return render_template('thumbnails.html',
                           thumbnails=thumbnails,
                           start_idx=start_idx,
                           total_pages=len(images),
                           has_next_page=has_next_page)



@app.route('/image/<int:image_index>')
def serve_image(image_index):
    """
    根据图片索引返回图片内容
    """
    if IMAGE_FOLDER is None or image_index < 0 or image_index >= len(images):
        return "Image index out of range.", 404

    image_path = os.path.join(IMAGE_FOLDER, images[image_index])
    try:
        return send_file(image_path)
    except FileNotFoundError:
        return "Image not found.", 404


@app.route('/view/<int:image_index>')
def view_image(image_index):
    """
    显示单张图片的详情页
    """
    global current_image_index
    current_image_index = image_index  # 更新当前查看的图片索引

    if IMAGE_FOLDER is None or image_index < 0 or image_index >= len(images):
        return "Image index out of range.", 404

    current_image = images[image_index]
    return render_template('viewer.html',
                           total=len(images),
                           index=image_index,
                           filename=current_image)


@app.route('/action', methods=['POST'])
def action():
    """
    处理表单提交的操作：保留、删除、下一张、上一张、下10张
    """
    global current_image_index, start_idx

    op = request.form.get('op')
    if op == 'delete':
        # 删除当前图片
        if IMAGE_FOLDER and 0 <= current_image_index < len(images):
            filename = images[current_image_index]
            image_path = os.path.join(IMAGE_FOLDER, filename)
            try:
                os.remove(image_path)
                print(f"[INFO] Deleted: {filename}")
            except OSError as e:
                print(f"[ERROR] Failed to delete {filename}: {e}")
            # 从列表中移除
            images.pop(current_image_index)
            # 如果删除的是最后一张，调整索引
            if current_image_index >= len(images):
                current_image_index = max(0, len(images) - 1)

    elif op == 'keep':
        # 保留图片
        pass

    elif op == 'next':
        # 下一张图片
        if current_image_index < len(images) - 1:
            current_image_index += 1

    elif op == 'prev':
        # 上一张图片
        if current_image_index > 0:
            current_image_index -= 1

    elif op == 'next_page':
        # 下 10 张
        if start_idx + images_per_page + 10 < len(images):
            start_idx += 10
    elif op == 'prev_page':  # 上一页
        if start_idx > 10:
            start_idx -= 10
        else:
            start_idx = 0
    return redirect(url_for('index'))

@app.route('/delete', methods=['POST'])
def delete_images():
    """
    批量删除选中的图片
    """
    global current_image_index, start_idx
    if not IMAGE_FOLDER:
        return jsonify({"error": "Invalid folder"}), 400

    indices = request.json.get('indices', [])
    deleted_files = []

    for index in sorted(indices, reverse=True):
        if 0 <= index < len(images):
            filename = images[index]
            image_path = os.path.join(IMAGE_FOLDER, filename)
            try:
                os.remove(image_path)
                deleted_files.append(filename)
            except OSError as e:
                print(f"[ERROR] Failed to delete {filename}: {e}")
                continue

            images.pop(index)  # 从列表中移除
    start_idx += 10 - len(deleted_files)
    print(start_idx)
    return jsonify({"deleted": deleted_files})

@app.route('/delete_page', methods=['POST'])
def delete_page():
    global current_image_index, start_idx
    deleted_files = []
    for index in range(10):
        if 0 <= start_idx < len(images):
            filename = images[start_idx]
            image_path = os.path.join(IMAGE_FOLDER, filename)
            try:
                os.remove(image_path)
                deleted_files.append(filename)
            except OSError as e:
                print(f"[ERROR] Failed to delete {filename}: {e}")
                continue

            images.pop(start_idx)  # 从列表中移除
    return jsonify({"deleted": deleted_files})
if __name__ == '__main__':
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
