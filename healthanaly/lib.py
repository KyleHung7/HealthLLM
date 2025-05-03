import nh3
import markdown

def mdToHtml(text):
  return nh3.clean(markdown.markdown(text))