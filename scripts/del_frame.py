import os
import subprocess

def open_in_vscode_preview(file_path):
    """
    调用 VSCode 'code -r' 打开文件。
    如果 VSCode 设置已启用预览模式( enablePreview = true )，
    那么再次打开新文件时会“替换”上一次的预览标签。
    """
    subprocess.Popen(["code", "-r", file_path])
    # 注意：Popen 非阻塞，不会卡住脚本。
    # 如果你想阻塞可以换 subprocess.run(["code", "-r", file_path])，
    # 但阻塞/非阻塞并不影响能否使用预览模式。

def manage_images(folder):
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    images.sort()
    if not images:
        print("No images found.")
        return
    
    for fn in images:
        full_path = os.path.join(folder, fn)
        print(f"Processing: {fn}")
        while True:
            cmd = input("[o]pen / [d]elete / [k]eep / [q]uit: ").strip().lower()
            if cmd == 'q':
                return
            elif cmd == 'o':
                open_in_vscode_preview(full_path)
                print("Opened in VSCode. (If you see italic tab, it's in preview mode.)")
                print("Opening another image in preview mode will auto-close/replace this tab.")
                continue
            elif cmd == 'd':
                os.remove(full_path)
                print(f"Deleted: {fn}")
                break
            elif cmd == 'k':
                print(f"Kept: {fn}")
                break
            else:
                print("Invalid input.")

def main():
    folder = "/data4/lzd/iccv25/vis/imgL/1"
    manage_images(folder)

if __name__ == "__main__":
    main()
