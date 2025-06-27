from clean import Clean
import os


# # 再帰的にすべてのファイルを探索
# for root, dirs, files in os.walk(base_dir):
#     for file in files:
#         if "cleaned" in file:
#             file_path = os.path.join(root, file)
#             try:
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#             except Exception as e:
#                 print(f"Failed to delete {file_path}: {e}")

base_dir = "../../data/aozora/"
output_base_dir = "../../data/aozora_cleaned/"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if "cleaned" not in file and file.endswith(".txt"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                cleaned_text = Clean.preprocess_text(text)

                relative_path = os.path.relpath(root, base_dir)
                output_dir = os.path.join(output_base_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_filename = f"{file[:-4]}_cleaned.txt"
                output_path = os.path.join(output_dir, output_filename)

                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                print(f"Cleaned and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
