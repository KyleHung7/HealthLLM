import nh3
import markdown
import os

def mdToHtml(text):
  return nh3.clean(markdown.markdown(text))

def clear_user_data_folder(user_id, folder_type):
  # 定義資料夾的路徑
  folder_path = f"static/{user_id}/{folder_type}"

  # 確認資料夾是否存在
  if os.path.exists(folder_path):
    # 刪除 summary 資料夾中的所有檔案
    for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)  # 刪除檔案或符號鏈結
      except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')
  else:
    print("The specified folder does not exist.")