import os
import threading
import pandas as pd
from flask import Flask, render_template, request, send_file, make_response, redirect, url_for
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_login import login_required, current_user
from auth import init_auth, get_user_upload_folder, load_user_settings
from datetime import datetime
from lib import mdToHtml, clear_user_data_folder

# Load environment variables
load_dotenv()

# Allow OAuth over HTTP for development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Flask and SocketIO initialization
app = Flask(__name__)
app.config['TMP_FOLDER'] = 'tmp'
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-secret-key")
app.config['GOOGLE_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")
app.config['OAUTHLIB_INSECURE_TRANSPORT'] = True

# Initialize authentication
init_auth(app)

socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

logLess = False
if logLess:
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print(" * Running on http://127.0.0.1:5000")

# Create upload and temporary folders
os.makedirs(app.config['TMP_FOLDER'], exist_ok=True)

# Onboarding page
@app.route('/onboarding')
@login_required
def onboarding():
    response = make_response(render_template('onboarding.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Home page
@app.route('/')
def index():
    if current_user.is_authenticated:
        user_settings = load_user_settings(current_user.id)
        account_role = user_settings.get('account_role')
        
        # Redirect to onboarding if account_role is not set
        if not account_role:
            return redirect(url_for('onboarding'))
        
        if account_role == 'elderly':
            return render_template('elderly_index.html')
    return render_template('index.html')

# Validate input values
def validate_health_data(data, data_type):
    if data_type == 'blood_pressure':
        fields = {
            'morning_systolic': (50, 250),
            'morning_diastolic': (30, 150),
            'morning_pulse': (30, 200),
            'evening_systolic': (50, 250),
            'evening_diastolic': (30, 150),
            'evening_pulse': (30, 200)
        }
    elif data_type == 'blood_sugar':
        fields = {
            'morning_fasting': (50, 300),
            'morning_postprandial': (70, 400),
            'evening_fasting': (50, 300),
            'evening_postprandial': (70, 400)
        }
    
    for field, (min_val, max_val) in fields.items():
        value = data.get(field, '無')
        if value != '無':
            try:
                num = float(value)
                if not (min_val <= num <= max_val):
                    return f"{field} 數值必須在 {min_val} 到 {max_val} 之間"
            except ValueError:
                return f"{field} 必須是有效數字"
    return None

# Background task for saving health data
def save_health_data_background(data_dict, data_type, user_folder, overwrite=False):
    try:
        # Convert form data to DataFrame
        data = {'Date': [data_dict['date']]}
        if data_type == 'blood_pressure':
            data.update({
                'Morning_Systolic': [data_dict.get('morning_systolic', '無')],
                'Morning_Diastolic': [data_dict.get('morning_diastolic', '無')],
                'Morning_Pulse': [data_dict.get('morning_pulse', '無')],
                'Evening_Systolic': [data_dict.get('evening_systolic', '無')],
                'Evening_Diastolic': [data_dict.get('evening_diastolic', '無')],
                'Evening_Pulse': [data_dict.get('evening_pulse', '無')]
            })
        elif data_type == 'blood_sugar':
            data.update({
                'Morning_Fasting': [data_dict.get('morning_fasting', '無')],
                'Morning_Postprandial': [data_dict.get('morning_postprandial', '無')],
                'Evening_Fasting': [data_dict.get('evening_fasting', '無')],
                'Evening_Postprandial': [data_dict.get('evening_postprandial', '無')]
            })
        
        df = pd.DataFrame(data)
        df.fillna("無", inplace=True)
        
        # Save to CSV
        csv_filename = f"{data_type}.csv"
        csv_path = os.path.join(user_folder, csv_filename)
        
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            if data_dict['date'] in existing_df['Date'].values and not overwrite:
                socketio.emit('confirm_overwrite', {
                    'message': f'🟡 {data_type.replace("_", " ")} 當日資料已存在，是否覆蓋？',
                    'data': data_dict,
                    'data_type': data_type,
                    'user_folder': user_folder,
                    'event_type': 'summary'
                })
                return
            elif overwrite:
                existing_df = existing_df[existing_df['Date'] != data_dict['date']]
                new_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                new_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            new_df = df
        new_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        socketio.emit('update', {'message': f'🟢 {data_type.replace("_", " ")}紀錄儲存成功', 'event_type': 'summary'})
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 紀錄儲存錯誤: {str(e)}", 'event_type': 'summary'})

# Background task for trend analysis
def trend_background_task(file_path, user_id):
    try:
        df = pd.read_csv(file_path)
        df.fillna("無", inplace=True)

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

# Save health data to CSV
@app.route('/save_health_data', methods=['POST'])
@login_required
def save_health_data():
    try:
        data = {
            'date': request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        }

        # Determine if it's blood pressure or blood sugar based on submitted form
        bp_fields = ['morning_systolic', 'morning_diastolic', 'morning_pulse', 
                    'evening_systolic', 'evening_diastolic', 'evening_pulse']
        sugar_fields = ['morning_fasting', 'morning_postprandial', 
                       'evening_fasting', 'evening_postprandial']

        is_bp = any(field in request.form for field in bp_fields)
        is_sugar = any(field in request.form for field in sugar_fields)

        if is_bp and not is_sugar:
            data_type = 'blood_pressure'
            for field in bp_fields:
                data[field] = request.form.get(field, '無')
        elif is_sugar and not is_bp:
            data_type = 'blood_sugar'
            for field in sugar_fields:
                data[field] = request.form.get(field, '無')
        else:
            return "請正確填寫血壓或血糖紀錄", 400

        # Validate input values
        validation_error = validate_health_data(data, data_type)
        if validation_error:
            socketio.emit('update', {'message': f"❌ 輸入錯誤: {validation_error}", 'event_type': 'summary'})
            return validation_error, 400

        # Get user folder in the main thread
        user_folder = get_user_upload_folder(current_user.id)

        socketio.emit('update', {'message': f'🟢 資料上傳成功，開始儲存{data_type.replace("_", " ")}紀錄...', 'event_type': 'summary'})
        thread = threading.Thread(target=save_health_data_background, args=(data, data_type, user_folder))
        thread.start()
        return '資料已儲存。', 200
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 資料處理錯誤: {str(e)}", 'event_type': 'summary'})
        return f'資料處理錯誤: {str(e)}', 500

# Handle overwrite confirmation
@app.route('/overwrite_health_data', methods=['POST'])
@login_required
def overwrite_health_data():
    try:
        data = request.form.get('data')
        data_type = request.form.get('data_type')
        user_folder = get_user_upload_folder(current_user.id)
        import json
        data_dict = json.loads(data)
        
        thread = threading.Thread(target=save_health_data_background, args=(data_dict, data_type, user_folder, True))
        thread.start()
        return '覆蓋資料已處理。', 200
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 覆蓋資料錯誤: {str(e)}", 'event_type': 'summary'})
        return f'覆蓋資料錯誤: {str(e)}', 500

# Background task for trend analysis
def trend_background_task(file_path, user_id):
    try:
        df = pd.read_csv(file_path)
        df.fillna("無", inplace=True)

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

# Upload CSV for trend analysis
@app.route('/upload_trend', methods=['POST'])
@login_required
def upload_trend():
    file = request.files.get('file')
    if not file or file.filename == '':
        return '請選擇檔案', 400

    filename = secure_filename(file.filename)
    user_folder = get_user_upload_folder(current_user.id)
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