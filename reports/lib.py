import markdown
import os
import shutil

def mdToHtml(markdown_text: str) -> str:
    """Converts markdown text to HTML."""
    return markdown.markdown(markdown_text)

def clear_user_data_folder(user_id: str, subfolder: str):
    """Clears and recreates a specific subfolder within a user's static data directory."""
    # 注意：這裡的路徑需要與 app.py 中的 UPLOAD_FOLDER 設定一致
    folder_path = os.path.join("static", "user_data", user_id, subfolder)
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Cleared folder: {folder_path}")
        except OSError as e:
            print(f"Error clearing folder {folder_path}: {e}")
    os.makedirs(folder_path, exist_ok=True) # Recreate the folder