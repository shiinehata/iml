import os

# Các thư mục hoặc tệp không muốn đưa vào (ví dụ: virtual environment, cache)
EXCLUDE_DIRS = {'__pycache__', '.venv', 'venv', '.git', '.vscode'}
EXCLUDE_FILES = {'export_project.py'} # Không cần đưa chính file này vào

def generate_project_context(root_dir, output_file):
    """
    Duyệt qua dự án, ghi cấu trúc thư mục và nội dung file vào một file duy nhất.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Bước 1: In cấu trúc thư mục trước
        f.write("CẤU TRÚC THƯ MỤC DỰ ÁN:\n")
        f.write("========================\n")
        for root, dirs, files in os.walk(root_dir):
            # Loại bỏ các thư mục không cần thiết
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            sub_indent = ' ' * 4 * (level + 1)
            for file_name in files:
                if file_name.endswith('.py') and file_name not in EXCLUDE_FILES:
                    f.write(f"{sub_indent}{file_name}\n")

        f.write("\n\nNỘI DUNG CHI TIẾT CÁC TỆP:\n")
        f.write("===========================\n\n")

        # Bước 2: In nội dung từng tệp
        for root, dirs, files in os.walk(root_dir):
            # Loại bỏ các thư mục không cần thiết
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file_name in files:
                if file_name.endswith('.py') and file_name not in EXCLUDE_FILES:
                    file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(file_path, root_dir)

                    f.write(f"--- START FILE: {relative_path} ---\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file_content:
                            f.write(file_content.read())
                    except Exception as e:
                        f.write(f"*** Không thể đọc file: {e} ***")
                    f.write(f"\n--- END FILE: {relative_path} ---\n\n")

if __name__ == "__main__":
    project_directory = '.'  # '.' nghĩa là thư mục hiện tại
    output_filename = 'project_context_for_gemini.txt'
    generate_project_context(project_directory, output_filename)
    print(f"Đã xuất toàn bộ dự án vào file: {output_filename}")