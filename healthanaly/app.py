import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from health_analysis import process_health_summary, health_trend_analysis, answer_care_question, default_prompt
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, async_mode='threading')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def background_task(file_path):
    try:
        html_content, pdf_path = process_health_summary(file_path, default_prompt)
        socketio.emit('update', {'message': '🟢 健康摘要生成完成'})
        socketio.emit('summary_result', {
            'html_content': html_content,
            'pdf_url': '/' + pdf_path if pdf_path else None
        })
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 摘要生成錯誤: {str(e)}"})

def trend_background_task(file_path):
    try:
        result = health_trend_analysis(file_path)
        socketio.emit('update', {'message': '🟢 趨勢分析完成'})
        socketio.emit('trend_result', {'trend_output': result})
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 趨勢分析錯誤: {str(e)}"})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_summary', methods=['POST'])
def upload_summary():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        socketio.emit('update', {'message': '🟢 檔案上傳成功，開始生成健康摘要...'})
        thread = threading.Thread(target=background_task, args=(file_path,))
        thread.start()
        return 'File uploaded and processing started.', 200

@app.route('/upload_trend', methods=['POST'])
def upload_trend():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        socketio.emit('update', {'message': '🟢 檔案上傳成功，開始趨勢分析...'})
        thread = threading.Thread(target=trend_background_task, args=(file_path,))
        thread.start()
        return 'File uploaded and processing started.', 200

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form.get('question', '')
    if not question.strip():
        return '請輸入問題', 400
    try:
        answer = answer_care_question(question)
        socketio.emit('question_result', {'answer': answer})
        return 'Question processed.', 200
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 問題回答錯誤: {str(e)}"})
        return 'Error processing question.', 500

if __name__ == '__main__':
    socketio.run(app, debug=True)