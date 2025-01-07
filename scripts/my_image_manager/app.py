from flask import Flask, request, send_file, render_template, redirect, url_for
import os
import sys

app = Flask(__name__)

# 全局变量：指定的图片路径、图片列表、当前图片索引
IMAGE_FOLDER = None
images = []
current_image_index = 0

@app.before_first_request
def load_images():
    """
    在第一次请求前加载指定路径下的图片文件，并按整数部分排序
    """
    global images
    valid_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    if IMAGE_FOLDER is None or not os.path.isdir(IMAGE_FOLDER):
        print(f"[ERROR] Invalid image path: {IMAGE_FOLDER}")
        return

    # 获取图片文件列表，并提取整数部分排序
    def extract_int(filename):
        """
        从文件名中提取整数部分用于排序
        假设文件名格式为 xxx.png，其中 xxx 是整数
        """
        try:
            return int(os.path.splitext(filename)[0][6:])  # 提取文件名（去掉扩展名）并转换为整数
        except ValueError:
            return float('inf')  # 如果无法转换为整数，放在最后

    images = [
        f for f in os.listdir(IMAGE_FOLDER)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    images.sort(key=extract_int)  # 按整数部分排序
    print(f"[INFO] Loaded {len(images)} images from {IMAGE_FOLDER}")

@app.route('/')
def index():
    """
    显示当前图片，并提供操作按钮
    """
    global current_image_index

    # 如果没有图片文件
    if not images:
        return "No images found in the specified folder."

    # 防止索引越界
    current_image_index = max(0, min(current_image_index, len(images) - 1))

    # 获取当前图片文件名
    current_image = images[current_image_index]

    return render_template('viewer.html',
                           total=len(images),
                           index=current_image_index,
                           filename=current_image)


@app.route('/image/<int:image_index>')
def serve_image(image_index):
    """
    根据图片索引返回图片内容
    """
    if image_index < 0 or image_index >= len(images):
        return "Image index out of range.", 404

    image_path = os.path.join(IMAGE_FOLDER, images[image_index])
    try:
        return send_file(image_path)
    except FileNotFoundError:
        return "Image not found.", 404


@app.route('/action', methods=['POST'])
def action():
    """
    处理表单提交的操作：保留、删除、下一张、上一张
    """
    global current_image_index

    op = request.form.get('op')
    if op == 'delete':
        # 删除当前图片
        if 0 <= current_image_index < len(images):
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

    return redirect(url_for('index'))


if __name__ == '__main__':
    # 从命令行参数读取图片路径和端口号
    if len(sys.argv) != 3:
        print("Usage: python app.py <image_folder_path> <port>")
        sys.exit(1)

    IMAGE_FOLDER = sys.argv[1]
    port = int(sys.argv[2])

    # 检查路径是否合法
    if not os.path.isdir(IMAGE_FOLDER):
        print(f"[ERROR] The specified path is not a valid directory: {IMAGE_FOLDER}")
        sys.exit(1)

    print(f"[INFO] Serving images from: {IMAGE_FOLDER}")
    app.run(host='0.0.0.0', port=port, debug=True)
