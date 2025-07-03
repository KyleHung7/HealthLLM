import nh3
import markdown
import os
import re

def mdToHtml(text):
  html = markdown.markdown(text)
  return nh3.clean(html)

def strip_html_tags(html_text):
  if not html_text:
    return ""
  clean = re.compile('<.*?>')
  text = re.sub(clean, '', html_text)
  text = text.replace(' ', ' ').replace('&', '&').replace('<', '<').replace('>', '>')
  return text

def clear_user_data_folder(user_id, folder_type):
  folder_path = f"static/users/{user_id}/{folder_type}"
  if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
      except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')
  else:
    print("The specified folder does not exist.")