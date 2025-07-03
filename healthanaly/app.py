import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import markdown
import requests

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, Response, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_socketio import SocketIO, emit, join_room
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.message import EmailMessage
import base64

from google_auth_oauthlib.flow import InstalledAppFlow
import health_analysis
import auth
from google_auth_oauthlib.flow import Flow
from auth import init_auth, get_user_upload_folder, load_user_settings, get_user_by_id
from img_recognition import img_recognition_bp
from lib import mdToHtml, strip_html_tags

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_for_development')
app.config['UPLOAD_FOLDER'] = 'static/users' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['GOOGLE_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")
app.config['OAUTHLIB_INSECURE_TRANSPORT'] = True
if os.getenv('TUNNEL_MODE') == "True":
    app.config['SERVER_NAME'] = os.getenv("SERVER_NAME")

RAG_SERVER_URL = os.getenv("RAG_SERVER_URL")
if not RAG_SERVER_URL:
    print("Warning: RAG_SERVER_URL not set in environment variables.")

init_auth(app)
app.register_blueprint(img_recognition_bp)

socketio = SocketIO(app)
user_sid_map = {}

# --- Helper functions ---
def get_user_data_path(user_id, subfolder=None, filename=None):
    base_dir = get_user_upload_folder(user_id)
    if subfolder:
        sub_dir = os.path.join(base_dir, subfolder)
        os.makedirs(sub_dir, exist_ok=True)
        if filename:
            return os.path.join(sub_dir, filename)
        return sub_dir
    if filename:
        return os.path.join(base_dir, filename)
    return base_dir

def clear_user_data_folder(user_id: str, subfolder: str):
    folder_path = get_user_data_path(user_id, subfolder)
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print(f"Error clearing folder {folder_path}: {e}")
    os.makedirs(folder_path, exist_ok=True)

# --- Authorization Helper ---
def is_authorized_for_user(target_user_id):
    if not current_user.is_authenticated:
        return False
    if str(target_user_id) == str(current_user.id):
        return True
    user_settings = load_user_settings(current_user.id)
    bound_accounts = user_settings.get('bound_accounts', [])
    return str(target_user_id) in bound_accounts

# --- Gmail API Function ---
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
CREDENTIALS_FILE = "gmail_credential.json"
TOKEN_FILE = "token.pickle"

def send_email_with_gmail_api(sender_email, recipient_email, subject, body, attachment_path=None):
    creds = None
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
    except (EOFError, pickle.PickleError) as e:
        print(f"Error loading {TOKEN_FILE}: {e}. Re-authenticating...")
        creds = None
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                return False, f"credentials.json not found at {CREDENTIALS_FILE}."
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)

    try:
        service = build('gmail', 'v1', credentials=creds)
        message = EmailMessage()
        message.set_content(body, subtype='html')
        message['To'] = recipient_email
        message['From'] = sender_email
        message['Subject'] = subject

        if attachment_path:
            if not os.path.exists(attachment_path):
                return False, f"Attachment file not found: {attachment_path}"
            with open(attachment_path, 'rb') as f:
                pdf_data = f.read()
            message.add_attachment(pdf_data, maintype='application', subtype='pdf', filename=os.path.basename(attachment_path))

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        send_message = service.users().messages().send(userId="me", body=create_message).execute()
        print(f"✅ Email sent, ID: {send_message['id']}")
        return True, "Email sent successfully."

    except HttpError as error:
        print(f"Error sending email: {error}")
        return False, f"Failed to send email: {error}"
    except Exception as e:
        print(f"Unknown error sending email: {e}")
        return False, f"Failed to send email: {e}"

# --- Health Data Logic ---
def save_health_data_to_csv(user_id, date, data_dict, data_type):
    user_folder = get_user_upload_folder(user_id)
    csv_filename = f"{data_type}.csv"
    csv_path = os.path.join(user_folder, csv_filename)

    if data_type == 'blood_pressure':
        columns = ['Date', 'Morning_Systolic', 'Morning_Diastolic', 'Morning_Pulse',
                   'Noon_Systolic', 'Noon_Diastolic', 'Noon_Pulse',
                   'Evening_Systolic', 'Evening_Diastolic', 'Evening_Pulse']
    elif data_type == 'blood_sugar':
        columns = ['Date', 'Morning_Fasting', 'Morning_Postprandial',
                   'Noon_Fasting', 'Noon_Postprandial',
                   'Evening_Fasting', 'Evening_Postprandial']
    else:
        raise ValueError("Invalid data_type specified")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    else:
        df = pd.DataFrame(columns=columns)

    for col in columns:
        if col != 'Date' and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if date in df['Date'].values:
        row_index = df[df['Date'] == date].index[0]
        for key, value in data_dict.items():
            numeric_value = pd.to_numeric(value, errors='coerce')
            df.loc[row_index, key] = numeric_value
    else:
        new_row = {'Date': date}
        for col in columns:
            if col != 'Date':
                value = data_dict.get(col)
                numeric_value = pd.to_numeric(value, errors='coerce')
                new_row[col] = numeric_value
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[columns]

    df.sort_values(by='Date', inplace=True)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    socketio.emit('update', {
        'message': f'🟢 {date} 的 {data_type.replace("_", " ")} 紀錄已更新',
        'event_type': 'summary'
    }, room=user_sid_map.get(user_id))

# --- Route definitions ---
@app.context_processor
def inject_user_role():
    if current_user.is_authenticated:
        user_settings = auth.load_user_settings(current_user.id)
        return dict(account_role=user_settings.get('account_role'))
    return dict(account_role=None)

@app.route('/')
def index():
    if current_user.is_authenticated:
        user_settings = auth.load_user_settings(current_user.id)
        account_role = user_settings.get('account_role')
        
        if not account_role:
            return redirect(url_for('onboarding'))
        
        if account_role == 'elderly':
            return render_template('elderly_index.html')
        
        if account_role == 'general':
            return redirect(url_for('general_dashboard'))

        return render_template('index.html') # Fallback
    
    return render_template('index.html')

@app.route('/general_dashboard')
@login_required
def general_dashboard():
    user_settings = auth.load_user_settings(current_user.id)
    account_role = user_settings.get('account_role')

    if not account_role:
        return redirect(url_for('onboarding'))

    if account_role == 'general':
        return render_template('general_user_dashboard.html')
    else:
        return redirect(url_for('index'))

@app.route('/onboarding')
@login_required
def onboarding():
    return render_template('onboarding.html')

@app.route('/rag_chat')
@login_required
def rag_chat():
    return render_template('rag_chat.html')

# --- API Routes ---
@app.route('/save_health_data', methods=['POST'])
@login_required
def save_health_data():
    target_user_id = request.form.get('target_user_id')
    
    if not is_authorized_for_user(target_user_id):
        return jsonify({'success': False, 'message': '權限不足，無法操作此帳戶。'}), 403
    
    date = request.form.get('date')
    if not date:
        return jsonify({'success': False, 'message': '缺少日期'}), 400

    bp_field_map = {
        'morning_systolic': 'Morning_Systolic', 'morning_diastolic': 'Morning_Diastolic', 'morning_pulse': 'Morning_Pulse',
        'noon_systolic': 'Noon_Systolic', 'noon_diastolic': 'Noon_Diastolic', 'noon_pulse': 'Noon_Pulse',
        'evening_systolic': 'Evening_Systolic', 'evening_diastolic': 'Evening_Diastolic', 'evening_pulse': 'Evening_Pulse'
    }
    sugar_field_map = {
        'morning_fasting': 'Morning_Fasting', 'morning_postprandial': 'Morning_Postprandial',
        'noon_fasting': 'Noon_Fasting', 'noon_postprandial': 'Noon_Postprandial',
        'evening_fasting': 'Evening_Fasting', 'evening_postprandial': 'Evening_Postprandial'
    }

    bp_data_to_save = {csv_col: request.form.get(form_name) for form_name, csv_col in bp_field_map.items() if form_name in request.form}
    sugar_data_to_save = {csv_col: request.form.get(form_name) for form_name, csv_col in sugar_field_map.items() if form_name in request.form}
    
    bp_data_to_save = {k: v for k, v in bp_data_to_save.items() if v}
    sugar_data_to_save = {k: v for k, v in sugar_data_to_save.items() if v}

    try:
        if bp_data_to_save:
            save_health_data_to_csv(target_user_id, date, bp_data_to_save, 'blood_pressure')
        if sugar_data_to_save:
            save_health_data_to_csv(target_user_id, date, sugar_data_to_save, 'blood_sugar')
        
        if not bp_data_to_save and not sugar_data_to_save:
             return jsonify({'success': False, 'message': '未提交任何有效數據。'}), 400

        return jsonify({'success': True, 'message': '數據已成功儲存'})
    except Exception as e:
        print(f"Error in save_health_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'伺服器錯誤: {e}'}), 500

@app.route('/get_linked_accounts')
@login_required
def get_linked_accounts():
    user_settings = load_user_settings(current_user.id)
    bound_account_ids = user_settings.get('bound_accounts', [])
    accounts_info = []
    for user_id in bound_account_ids:
        user = get_user_by_id(user_id)
        if user:
            accounts_info.append({'id': user.id, 'name': user.name})
    return jsonify({'accounts': accounts_info})

@app.route('/api/get_health_data_for_date', methods=['GET'])
@login_required
def get_health_data_for_date():
    target_user_id = request.args.get('user_id', current_user.id)
    
    if not is_authorized_for_user(target_user_id):
        return jsonify({"error": "權限不足"}), 403
    
    selected_date_str = request.args.get('date')
    if not selected_date_str:
        return jsonify({"error": "Date parameter is required"}), 400

    data_to_send = {}
    
    for data_type in ['blood_pressure', 'blood_sugar']:
        csv_file = get_user_data_path(target_user_id, filename=f'{data_type}.csv')
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                date_row = df[df['Date'] == selected_date_str]
                if not date_row.empty:
                    row_dict = date_row.iloc[0].to_dict()
                    for csv_col, value in row_dict.items():
                        if csv_col != 'Date' and pd.notna(value):
                            form_field_name = csv_col.lower()
                            data_to_send[form_field_name] = str(int(value))
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    return jsonify(data_to_send)

@app.route('/api/check_bp_status', methods=['POST'])
def check_bp_status():
    data = request.get_json()
    systolic = data.get('systolic')
    diastolic = data.get('diastolic')
    pulse = data.get('pulse')
    status, advice, normal_range_info = health_analysis.analyze_blood_pressure(systolic, diastolic, pulse)
    return jsonify({'status': status, 'advice': advice, 'normal_range_info': normal_range_info})

@app.route('/api/check_bs_status', methods=['POST'])
def check_bs_status():
    data = request.get_json()
    value = data.get('value')
    measurement_type = data.get('type')
    status, advice, normal_range_info = health_analysis.analyze_blood_sugar(value, measurement_type)
    return jsonify({'status': status, 'advice': advice, 'normal_range_info': normal_range_info})

@app.route('/analyze_account_trend', methods=['POST'])
@login_required
def analyze_account_trend():
    target_user_id = request.form.get('user_id')
    if not is_authorized_for_user(target_user_id):
        return jsonify({'success': False, 'message': '權限不足'}), 403

    time_period = request.form.get('time_period')
    data_type = request.form.get('data_type')

    if data_type == 'blood_pressure':
        csv_file_path = get_user_data_path(target_user_id, filename='blood_pressure.csv')
    elif data_type == 'blood_sugar':
        csv_file_path = get_user_data_path(target_user_id, filename='blood_sugar.csv')
    else:
        return jsonify({'success': False, 'message': '無效的數據類型。'}), 400

    if not os.path.exists(csv_file_path):
        return jsonify({'success': False, 'message': '該數據類型無歷史紀錄可供分析。'}), 404
    
    try:
        trend_output_text, _, plotly_data_string, _ = \
            health_analysis.health_trend_analysis(csv_file_path, None, None, time_period, data_type, generate_pdf=False)
        
        if "錯誤" in trend_output_text:
            return jsonify({'success': False, 'message': trend_output_text}), 500

        trend_output_html = markdown.markdown(trend_output_text)

        report_params = {
            "data_type": data_type,
            "time_period": time_period,
            "user_id": target_user_id
        }
        
        response_data_partial = {
            'success': True,
            'message': '帳戶數據趨勢分析完成。',
            'trend_output_html': trend_output_html, 
            'report_params': report_params
        }
        
        partial_json = json.dumps(response_data_partial)
        final_json_string = partial_json[:-1] + f', "plot_data": {plotly_data_string}' + '}'
        
        return Response(final_json_string, mimetype='application/json')

    except Exception as e:
        error_message = f"分析帳戶數據趨勢失敗: {e}"
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': error_message}), 500

@app.route('/download_report', methods=['POST'])
@login_required
def download_report():
    params = request.get_json()
    target_user_id = params.get('user_id')
    data_type = params.get('data_type')
    time_period = params.get('time_period')

    if not is_authorized_for_user(target_user_id):
        return jsonify({'success': False, 'message': '權限不足'}), 403

    if data_type == 'blood_pressure':
        csv_file_path = get_user_data_path(target_user_id, filename='blood_pressure.csv')
    elif data_type == 'blood_sugar':
        csv_file_path = get_user_data_path(target_user_id, filename='blood_sugar.csv')
    else:
        return jsonify({'success': False, 'message': '無效的數據類型'}), 400

    if not os.path.exists(csv_file_path):
        return jsonify({'success': False, 'message': '找不到數據檔案'}), 404

    clear_user_data_folder(target_user_id, 'reports')
    analysis_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        base_output_dir = get_user_data_path(target_user_id)
        _, pdf_report_rel_static_path, _, pdf_filename = \
            health_analysis.health_trend_analysis(csv_file_path, base_output_dir, analysis_timestamp_str, time_period, data_type, generate_pdf=True)

        if not pdf_report_rel_static_path:
            return jsonify({'success': False, 'message': 'PDF 報告生成失敗'}), 500
        
        return send_from_directory('static', pdf_report_rel_static_path, as_attachment=True, download_name=pdf_filename)

    except Exception as e:
        print(f"下載報告時發生錯誤: {e}")
        return jsonify({'success': False, 'message': f'生成下載報告時發生錯誤: {e}'}), 500

@app.route('/generate_and_send_report', methods=['POST'])
@login_required
def generate_and_send_report():
    data = request.get_json()
    target_user_id = data.get('user_id')
    recipient_email = data.get('email')
    period = data.get('period')
    data_type = data.get('data_type')
    
    if not is_authorized_for_user(target_user_id):
        return jsonify({'success': False, 'message': '權限不足'}), 403

    if not recipient_email:
        return jsonify({'success': False, 'message': '請提供收件人電子郵件。'}), 400

    if data_type == 'blood_pressure':
        csv_file_path = get_user_data_path(target_user_id, filename='blood_pressure.csv')
    elif data_type == 'blood_sugar':
        csv_file_path = get_user_data_path(target_user_id, filename='blood_sugar.csv')
    else:
        return jsonify({'success': False, 'message': '無效的數據類型。'}), 400

    if not os.path.exists(csv_file_path):
        return jsonify({'success': False, 'message': '該數據類型無歷史紀錄可供生成報告。'}), 404

    clear_user_data_folder(target_user_id, 'reports')
    base_output_dir = get_user_data_path(target_user_id)
    analysis_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        _, pdf_report_rel_static_path, _, pdf_filename = \
            health_analysis.health_trend_analysis(csv_file_path, base_output_dir, analysis_timestamp_str, period, data_type, generate_pdf=True)
        if not pdf_report_rel_static_path:
            return jsonify({'success': False, 'message': '郵寄時 PDF 報告生成失敗'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'郵寄時生成報告失敗: {e}'}), 500

    try:
        report_subject = pdf_filename.replace('.pdf', '')
        report_body = f"您好，<br><br>這是您在 HealthLLM 系統中為帳戶 {get_user_by_id(target_user_id).name} 生成的健康趨勢報告。<br><br>請查收附件。<br><br>此致，<br>HealthLLM 團隊"
        abs_pdf_path = os.path.abspath(os.path.join('static', pdf_report_rel_static_path))

        success, message = send_email_with_gmail_api(
            sender_email="healthllm.team@gmail.com",
            recipient_email=recipient_email,
            subject=report_subject,
            body=report_body,
            attachment_path=abs_pdf_path
        )

        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message}), 500
    except Exception as e:
        print(f"Error sending report: {e}")
        return jsonify({'success': False, 'message': f'寄送報告時發生錯誤: {e}'}), 500

# --- RAG Chat Routes ---
@app.route('/rag_submit', methods=['POST'])
@login_required
def rag_submit():
    question = request.form.get('question', '').strip()
    voice_mode = request.form.get('voice_mode') == 'true'
    print("Asking question: ", question, " Voice mode: ", voice_mode)

    if not question:
        return jsonify({'error': '請輸入問題'}), 400

    if not RAG_SERVER_URL:
        return jsonify({'error': 'RAG 伺服器地址未設定'}), 500

    try:
        # The payload for the /submit endpoint is JSON.
        payload = {
            'question': question,
            'voice_mode': voice_mode
        }
        
        # We only need to call the /submit endpoint. The RAG server will handle TTS internally.
        chat_response = requests.post(f"{RAG_SERVER_URL}/submit", json=payload)
        chat_response.raise_for_status()
        
        rag_response_json = chat_response.json()
        
        rag_answer_md = rag_response_json.get('answer', '無法取得回答')
        audio_url = rag_response_json.get('audio', None)
        
        rag_answer_html = mdToHtml(rag_answer_md)

        return jsonify({'answer': rag_answer_html, 'audio': audio_url})

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error communicating with RAG server: {e}")
        return jsonify({'error': f'與 RAG 伺服器通訊錯誤: {e}'}), 500
    except Exception as e:
        app.logger.error(f"Error processing RAG submit: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'處理 RAG 提交錯誤: {e}'}), 500

@app.route('/rag_record', methods=['POST'])
@login_required
def rag_record():
    audio_file = request.files.get('audio')
    mode = request.form.get('mode')

    if not audio_file:
        return jsonify({'error': '未收到音訊檔案'}), 400

    if not RAG_SERVER_URL:
        return jsonify({'error': 'RAG 伺服器地址未設定'}), 500

    try:
        files = {
            'audio': (audio_file.filename, audio_file.stream, audio_file.mimetype),
            'mode': (None, mode)
        }
        stt_response = requests.post(f"{RAG_SERVER_URL}/record", files=files)
        stt_response.raise_for_status()
        rag_response_json = stt_response.json()
        
        transcription = rag_response_json.get('transcription', '無法轉錄')
        rag_answer_html = mdToHtml(rag_response_json.get('answer', ''))
        audio_url = rag_response_json.get('audio', None)

        return jsonify({'transcription': transcription, 'answer': rag_answer_html, 'audio': audio_url})

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error communicating with RAG server: {e}")
        return jsonify({'error': f'與 RAG 伺服器通訊錯誤: {e}'}), 500
    except Exception as e:
        app.logger.error(f"Error processing RAG record: {e}")
        return jsonify({'error': f'處理 RAG 錄音錯誤: {e}'}), 500

@app.route('/audio/<filename>')
@login_required
def serve_rag_audio(filename):
    if not RAG_SERVER_URL:
        app.logger.error('RAG 伺服器地址未設定')
        return send_file('static/error_audio.mp3', mimetype='audio/mpeg')
    try:
        rag_audio_url = f"{RAG_SERVER_URL}/audio/{filename}"
        response = requests.get(rag_audio_url, stream=True)
        response.raise_for_status()
        return Response(response.iter_content(chunk_size=1024), status=response.status_code, mimetype=response.headers.get('Content-Type', 'audio/mpeg'))
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching audio from RAG server: {e}")
        return send_file('static/error_audio.mp3', mimetype='audio/mpeg')

# --- Socket.IO & Gmail Auth ---
@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        user_id = current_user.id
        user_sid_map[user_id] = request.sid
        join_room(user_id)
        print(f'Client connected: {request.sid} for user {user_id}')

@socketio.on('disconnect')
def handle_disconnect():
    disconnected_sid = request.sid
    user_id_to_remove = next((user_id for user_id, sid in user_sid_map.items() if sid == disconnected_sid), None)
    if user_id_to_remove:
        user_sid_map.pop(user_id_to_remove, None)
        print(f'Client disconnected: {disconnected_sid} for user {user_id_to_remove}')

def get_gmail_auth_flow():
    return Flow.from_client_secrets_file(CREDENTIALS_FILE, scopes=SCOPES, redirect_uri=url_for('gmail_callback', _external=True))

@app.route('/authorize_gmail')
@login_required
def authorize_gmail():
    try:
        flow = get_gmail_auth_flow()
        auth_url, _ = flow.authorization_url(access_type='offline', include_granted_scopes='true', prompt='consent')
        session['gmail_flow_state'] = flow.state
        return redirect(auth_url)
    except Exception as e:
        return f"⚠️ 無法建立授權流程: {e}", 500

@app.route('/gmail_callback')
@login_required
def gmail_callback():
    state = session.get('gmail_flow_state')
    if not state or state != request.args.get('state'):
        return "State mismatch. Possible CSRF attack.", 400
    try:
        flow = get_gmail_auth_flow()
        flow.fetch_token(authorization_response=request.url)
        with open(TOKEN_FILE, 'wb') as token_file:
            pickle.dump(flow.credentials, token_file)
        return '✅ Gmail 授權成功！<a href="/">回到首頁</a>'
    except Exception as e:
        return f"❌ 授權流程錯誤: {e}", 500

# --- Run the application ---
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)