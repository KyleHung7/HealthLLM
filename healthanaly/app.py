import os
import threading
import pandas as pd
from flask import Flask, render_template, request, send_file, make_response
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_login import login_required, current_user
from health_analysis import (
    process_health_summary,
    health_trend_analysis,
    answer_care_question,
    validate_bp_csv,
    validate_sugar_csv,
    generate_html,
    generate_pdf_from_html
)
from auth import init_auth, get_user_upload_folder
from datetime import datetime
from lib import mdToHtml, clear_user_data_folder

# Load environment variables
load_dotenv()

# Allow OAuth over HTTP for development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Flask and SocketIO initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['TMP_FOLDER'] = 'tmp'
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-secret-key")
app.config['GOOGLE_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")
app.config['OAUTHLIB_INSECURE_TRANSPORT'] = True

# 初始化認證
init_auth(app)

socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

logLess = False
if logLess:
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print(" * Running on http://127.0.0.1:5000")

# Create upload and temporary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TMP_FOLDER'], exist_ok=True)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Background task for health summary
def summary_background_task(file_path, data_type, user_id):
    try:
        df = pd.read_csv(file_path)
        df.fillna("無", inplace=True)
        summary_df = process_health_summary(df, data_type)
        html_content = generate_html(summary_df, f"{data_type.replace('_', ' ').title()} 健康紀錄分析")
        
        # 使用使用者 ID 來命名 PDF 檔案
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        pdf_filename = f"{data_type}_{timestamp}_summary.pdf"
        generate_pdf_from_html(html_content, user_id, pdf_filename)
        
        socketio.emit('update', {'message': '🟢 健康摘要生成完成', 'event_type': 'summary'})
        socketio.emit('summary_result', {
            'html_content': html_content,
            'pdf_url': f'/download_pdf/{user_id}/{data_type}/{timestamp}',
            'event_type': 'summary'
        })
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 摘要生成錯誤: {str(e)}", 'event_type': 'summary'})

# Background task for trend analysis
def trend_background_task(file_path, user_id):
    try:
        df = pd.read_csv(file_path)
        df.fillna("無", inplace=True)

        # Determine data_type based on CSV validation
        data_type = 'blood_pressure' if validate_bp_csv(df) else 'blood_sugar' if validate_sugar_csv(df) else None
        if not data_type:
            socketio.emit('update', {'message': '❌ CSV 檔案格式不符合血壓或血糖分析要求', 'event_type': 'trend'})
            return

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        clear_user_data_folder(user_id, "trend")
        result = health_trend_analysis(file_path, user_id, timestamp)
        socketio.emit('update', {'message': '🟢 趨勢分析完成', 'event_type': 'trend'})
        socketio.emit('trend_result', {
            'trend_output': result,
            'trend_url': f'/download_trend/{user_id}/{data_type}/{timestamp}',
            'event_type': 'trend'
        })
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 趨勢分析錯誤: {str(e)}", 'event_type': 'trend'})

# Upload CSV for health summary and generate PDF
@app.route('/upload_summary', methods=['POST'])
@login_required
def upload_summary():
    file = request.files.get('file')
    if not file or file.filename == '':
        return '請選擇檔案', 400

    filename = secure_filename(file.filename)
    user_folder = get_user_upload_folder()
    file_path = os.path.join(app.config['TMP_FOLDER'], filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)
        df.fillna("無", inplace=True)
        if validate_bp_csv(df):
            data_type = 'blood_pressure'
        elif validate_sugar_csv(df):
            data_type = 'blood_sugar'
        else:
            return "CSV 欄位不符合血壓或血糖摘要格式", 400

        socketio.emit('update', {'message': '🟢 檔案上傳成功，開始生成健康摘要...', 'event_type': 'summary'})
        thread = threading.Thread(target=summary_background_task, args=(file_path, data_type, current_user.id))
        thread.start()
        return '檔案已上傳並開始處理。', 200
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 檔案處理錯誤: {str(e)}", 'event_type': 'summary'})
        return f'檔案處理錯誤: {str(e)}', 500

# Upload CSV for trend analysis
@app.route('/upload_trend', methods=['POST'])
@login_required
def upload_trend():
    file = request.files.get('file')
    if not file or file.filename == '':
        return '請選擇檔案', 400

    filename = secure_filename(file.filename)
    user_folder = get_user_upload_folder()
    file_path = os.path.join(user_folder, filename)
    file.save(file_path)

    socketio.emit('update', {'message': '🟢 檔案上傳成功，開始趨勢分析...', 'event_type': 'trend'})
    thread = threading.Thread(target=trend_background_task, args=(file_path, current_user.id))
    thread.start()
    return '檔案已上傳並開始處理。', 200

# Download PDF report
@app.route('/download_pdf/<user_id>/<data_type>/<timestamp>')
@login_required
def download_pdf(user_id, data_type, timestamp):
    print(f"Download pdf: {current_user.id}")
    # 確認使用者只能下載自己的文件
    if current_user.id != user_id:
        return '您沒有權限存取此檔案', 403
    pdf_filename = f"{data_type}_{timestamp}_summary.pdf"
    pdf_path = f"static/{user_id}/summary/{pdf_filename}"
    if os.path.exists(pdf_path):
        response = make_response(send_file(pdf_path, as_attachment=True, download_name=pdf_filename))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return 'PDF 文件不存在', 404

# Download trend image
@app.route('/download_trend/<user_id>/<data_type>/<timestamp>')
@login_required
def download_trend(user_id, data_type, timestamp):
    print(f"Download trend: {current_user.id}")
    # 確認使用者只能下載自己的文件
    if current_user.id != user_id:
        return '您沒有權限存取此檔案', 403
        
    if data_type == 'blood_pressure':
        image_path = f"static/{user_id}/trend/bp_trend_{timestamp}.png"
        download_name = f"bp_trend_{timestamp}.png"
    elif data_type == 'blood_sugar':
        image_path = f"static/{user_id}/trend/sugar_trend_{timestamp}.png"
        download_name = f"sugar_trend_{timestamp}.png"
    else:
        return '無效的趨勢圖型', 400
    
    if os.path.exists(image_path):
        response = make_response(send_file(image_path, as_attachment=True, download_name=download_name))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return '趨勢圖不存在', 404

# Answer caregiver questions
@app.route('/ask_question', methods=['POST'])
@login_required
def ask_question():
    question = request.form.get('question', '').strip()
    if not question:
        return '請輸入問題', 400
    try:
        answer = answer_care_question(question)
        answer_html = mdToHtml(answer)
        socketio.emit('question_result', {'answer': answer_html, 'event_type': 'question'})
        return '問題已處理', 200
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 問題回答錯誤: {str(e)}", 'event_type': 'question'})
        return '問題處理錯誤', 500

# Start server
if __name__ == '__main__':
    socketio.run(app, debug=True)
