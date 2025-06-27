import os
import zipfile

# 対象ディレクトリ（カレントディレクトリ）
target_dir = "."

for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".zip"):
            zip_path = os.path.join(root, file)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for zip_info in zf.infolist():
                        if not zip_info.is_dir():
                            txt_name = zip_info.filename
                            try:
                                text = zf.read(txt_name).decode('shift_jis', errors='ignore')
                                output_filename = f"{os.path.splitext(file)[0]}_{txt_name.replace('/', '_')}.txt"
                                output_path = os.path.join(root, output_filename)
                                with open(output_path, "w", encoding="utf-8") as out_f:
                                    out_f.write(text)
                                print(f"Converted {txt_name} in {file} to {output_filename}")
                            except Exception as e:
                                print(f"Failed to extract {txt_name} from {file}: {e}")
            except Exception as e:
                print(f"Failed to open zip file {zip_path}: {e}")