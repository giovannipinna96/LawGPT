# HELPER FUNCTION
import os
import shutil


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def copy_json_files(source_dir, dest_dir, verbose=False):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".json"):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(source_path, dest_path)
                print(f"File copied: {source_path} -> {dest_path}") if verbose else None
