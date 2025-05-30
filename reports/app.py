import os
import shutil
import pandas as pd
import numpy as np # Import numpy for np.integer check
from datetime import datetime, timedelta
import json
import pickle # For token.pickle

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import Gmail API related libraries
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.message import EmailMessage # Changed from email.mime.text/multipart/base
import base64

from google_auth_oauthlib.flow import InstalledAppFlow
# Import your health analysis module
import health_analysis

# Import auth module
import auth

from google_auth_oauthlib.flow import Flow


load_dotenv() # Load environment variables from .env

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_for_development')
app.config['UPLOAD_FOLDER'] = 'static/user_data' # Root directory for user data and uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload size

socketio = SocketIO(app)

# Initialize auth module (including Flask-Login and Authlib OAuth)
auth.init_auth(app)

# Map user ID to their Socket.IO session ID
user_sid_map = {}

# --- Helper functions ---
def mdToHtml(markdown_text: str) -> str:
    """Converts markdown text to HTML."""
    import markdown
    return markdown.markdown(markdown_text)

def clear_user_data_folder(user_id: str, subfolder: str):
    """Clears and recreates a specific subfolder within a user's static data directory."""
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(user_id), subfolder)
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Cleared folder: {folder_path}")
        except OSError as e:
            print(f"Error clearing folder {folder_path}: {e}")
    os.makedirs(folder_path, exist_ok=True) # Recreate the folder

def get_user_data_path(user_id, subfolder=None, filename=None):
    # Use secure_filename to sanitize user_id
    user_id_safe = secure_filename(user_id)
    base_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id_safe)
    os.makedirs(base_dir, exist_ok=True) # Ensure user root folder exists

    if subfolder:
        sub_dir = os.path.join(base_dir, subfolder)
        os.makedirs(sub_dir, exist_ok=True)
        if filename:
            return os.path.join(sub_dir, filename)
        return sub_dir
    
    if filename:
        return os.path.join(base_dir, filename)
    return base_dir

# --- Gmail API Email Sending Function (Updated to match example) ---
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
CREDENTIALS_FILE = "gmail_credential.json" # Your credentials file path
TOKEN_FILE = "token.pickle" # Global token file

def send_email_with_gmail_api(sender_email, recipient_email, subject, body, attachment_path=None):
    """Sends an email with optional attachment using Gmail API."""
    creds = None
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
    except (EOFError, pickle.PickleError) as e:
        print(f"Error loading {TOKEN_FILE}: {e}. Re-authenticating...")
        creds = None
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE) # Delete corrupted file

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                return False, f"credentials.json not found at {CREDENTIALS_FILE}. Please download it from Google Cloud Console."
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)

    try:
        service = build('gmail', 'v1', credentials=creds)
        message = EmailMessage() # Use EmailMessage from email.message
        message.set_content(body, subtype='html') # Set content as HTML
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


# --- Route definitions ---

@app.route('/')
def index():
    today_date = datetime.now().strftime('%Y-%m-%d')
    account_role = current_user.role if current_user.is_authenticated else None
    return render_template('index.html', today_date=today_date, current_user=current_user, account_role=account_role)

# auth.py handles /auth/login and /auth/logout routes

@app.route('/general_dashboard')
@login_required
def general_dashboard():
    return "<h1>已連結帳戶儀表板</h1><p>此功能待開發。</p><a href=\"/\">回首頁</a>"

@app.route('/rag_chat')
@login_required
def rag_chat():
    return "<h1>健康問答</h1><p>此功能待開發。</p><a href=\"/\">回首頁</a>"


@app.route('/api/get_today_health_data', methods=['GET'])
@login_required
def get_today_health_data():
    user_id = current_user.id
    bp_file = get_user_data_path(user_id, 'data', 'blood_pressure_records.csv')
    sugar_file = get_user_data_path(user_id, 'data', 'blood_sugar_records.csv')
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    data = {}

    # Read BP data
    if os.path.exists(bp_file):
        try:
            df_bp = pd.read_csv(bp_file, encoding='utf-8-sig')
            df_bp['Date'] = pd.to_datetime(df_bp['Date']).dt.strftime('%Y-%m-%d')
            today_bp = df_bp[df_bp['Date'] == today_str].iloc[0] if not df_bp[df_bp['Date'] == today_str].empty else None
            if today_bp is not None:
                for prefix in ['Morning', 'Midday', 'Evening']:
                    systolic = today_bp.get(f'{prefix}_Systolic', '')
                    diastolic = today_bp.get(f'{prefix}_Diastolic', '')
                    pulse = today_bp.get(f'{prefix}_Pulse', '')

                    # Convert numpy.int64 to standard Python int/float or empty string
                    data[f'{prefix.lower()}_systolic'] = int(systolic) if pd.notna(systolic) and isinstance(systolic, (int, float, np.integer)) else ''
                    data[f'{prefix.lower()}_diastolic'] = int(diastolic) if pd.notna(diastolic) and isinstance(diastolic, (int, float, np.integer)) else ''
                    data[f'{prefix.lower()}_pulse'] = int(pulse) if pd.notna(pulse) and isinstance(pulse, (int, float, np.integer)) else ''
        except Exception as e:
            print(f"Error reading BP file for today: {e}")

    # Read Sugar data
    if os.path.exists(sugar_file):
        try:
            df_sugar = pd.read_csv(sugar_file, encoding='utf-8-sig')
            df_sugar['Date'] = pd.to_datetime(df_sugar['Date']).dt.strftime('%Y-%m-%d')
            today_sugar = df_sugar[df_sugar['Date'] == today_str].iloc[0] if not df_sugar[df_sugar['Date'] == today_str].empty else None
            if today_sugar is not None:
                for prefix in ['Morning', 'Midday', 'Evening']:
                    fasting = today_sugar.get(f'{prefix}_Fasting', '')
                    postprandial = today_sugar.get(f'{prefix}_Postprandial', '')

                    # Convert numpy.int64 to standard Python int/float or empty string
                    data[f'{prefix.lower()}_fasting'] = int(fasting) if pd.notna(fasting) and isinstance(fasting, (int, float, np.integer)) else ''
                    data[f'{prefix.lower()}_postprandial'] = int(postprandial) if pd.notna(postprandial) and isinstance(postprandial, (int, float, np.integer)) else ''
        except Exception as e:
            print(f"Error reading Sugar file for today: {e}")
    
    return jsonify(data)


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

@app.route('/save_health_data', methods=['POST'])
@login_required
def save_health_data():
    user_id = current_user.id
    date = request.form.get('date')
    
    time_slot_map = {
        'morning': 'Morning',
        'noon': 'Midday',
        'evening': 'Evening'
    }
    
    is_bp_data = any(key.endswith('_systolic') or key.endswith('_diastolic') or key.endswith('_pulse') for key in request.form)
    is_sugar_data = any(key.endswith('_fasting') or key.endswith('_postprandial') for key in request.form)

    if not date:
        return jsonify({'success': False, 'message': '請選擇日期。'}), 400

    user_socket_sid = user_sid_map.get(user_id)

    if is_bp_data:
        bp_file_path = get_user_data_path(user_id, 'data', 'blood_pressure_records.csv')
        try:
            if os.path.exists(bp_file_path):
                df_bp = pd.read_csv(bp_file_path, encoding='utf-8-sig')
                df_bp['Date'] = pd.to_datetime(df_bp['Date']).dt.strftime('%Y-%m-%d')
            else:
                df_bp = pd.DataFrame(columns=['Date', 'Morning_Systolic', 'Morning_Diastolic', 'Morning_Pulse',
                                            'Midday_Systolic', 'Midday_Diastolic', 'Midday_Pulse',
                                            'Evening_Systolic', 'Evening_Diastolic', 'Evening_Pulse'])
            
            if date in df_bp['Date'].values:
                idx = df_bp[df_bp['Date'] == date].index[0]
            else:
                idx = len(df_bp)
                df_bp.loc[idx, 'Date'] = date

            for slot_key, mapped_slot in time_slot_map.items():
                systolic = request.form.get(f'{slot_key}_systolic')
                diastolic = request.form.get(f'{slot_key}_diastolic')
                pulse = request.form.get(f'{slot_key}_pulse')
                
                if systolic: df_bp.loc[idx, f'{mapped_slot}_Systolic'] = systolic
                if diastolic: df_bp.loc[idx, f'{mapped_slot}_Diastolic'] = diastolic
                if pulse: df_bp.loc[idx, f'{mapped_slot}_Pulse'] = pulse
            
            df_bp.to_csv(bp_file_path, index=False, encoding='utf-8-sig')
            if user_socket_sid:
                socketio.emit('update', {'event_type': 'summary', 'message': f'🟢 血壓紀錄已儲存於 {date}。'}, room=user_socket_sid)
            return jsonify({'success': True, 'message': '血壓紀錄已成功儲存。'})
        except Exception as e:
            if user_socket_sid:
                socketio.emit('update', {'event_type': 'summary', 'message': f'❌ 儲存血壓紀錄失敗: {e}'}, room=user_socket_sid)
            return jsonify({'success': False, 'message': f'儲存血壓紀錄失敗: {e}'}), 500

    elif is_sugar_data:
        sugar_file_path = get_user_data_path(user_id, 'data', 'blood_sugar_records.csv')
        try:
            if os.path.exists(sugar_file_path):
                df_sugar = pd.read_csv(sugar_file_path, encoding='utf-8-sig')
                df_sugar['Date'] = pd.to_datetime(df_sugar['Date']).dt.strftime('%Y-%m-%d')
            else:
                df_sugar = pd.DataFrame(columns=['Date', 'Morning_Fasting', 'Morning_Postprandial',
                                                'Midday_Fasting', 'Midday_Postprandial',
                                                'Evening_Fasting', 'Evening_Postprandial'])
            
            if date in df_sugar['Date'].values:
                idx = df_sugar[df_sugar['Date'] == date].index[0]
            else:
                idx = len(df_sugar)
                df_sugar.loc[idx, 'Date'] = date

            for slot_key, mapped_slot in time_slot_map.items():
                fasting = request.form.get(f'{slot_key}_fasting')
                postprandial = request.form.get(f'{slot_key}_postprandial')
                
                if fasting: df_sugar.loc[idx, f'{mapped_slot}_Fasting'] = fasting
                if postprandial: df_sugar.loc[idx, f'{mapped_slot}_Postprandial'] = postprandial
            
            df_sugar.to_csv(sugar_file_path, index=False, encoding='utf-8-sig')
            if user_socket_sid:
                socketio.emit('update', {'event_type': 'summary', 'message': f'🟢 血糖紀錄已儲存於 {date}。'}, room=user_socket_sid)
            return jsonify({'success': True, 'message': '血糖紀錄已成功儲存。'})
        except Exception as e:
            if user_socket_sid:
                socketio.emit('update', {'event_type': 'summary', 'message': f'❌ 儲存血糖紀錄失敗: {e}'}, room=user_socket_sid)
            return jsonify({'success': False, 'message': f'儲存血糖紀錄失敗: {e}'}), 500
    
    return jsonify({'success': False, 'message': '未收到有效的健康數據。'}), 400


@app.route('/upload_trend', methods=['POST'])
@login_required
def upload_trend():
    user_id = current_user.id
    user_socket_sid = user_sid_map.get(user_id)

    if 'file' not in request.files:
        if user_socket_sid:
            socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': '❌ 沒有檔案部分'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': '沒有檔案部分'}), 400
    file = request.files['file']
    if file.filename == '':
        if user_socket_sid:
            socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': '❌ 沒有選擇檔案'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': '沒有選擇檔案'}), 400
    
    data_type = request.form.get('data_type')
    time_period = request.form.get('time_period')

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        upload_path = get_user_data_path(user_id, 'data', filename)
        file.save(upload_path)
        
        clear_user_data_folder(user_id, 'trend')
        clear_user_data_folder(user_id, 'pca')
        clear_user_data_folder(user_id, 'reports')

        analysis_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            base_output_dir = get_user_data_path(user_id)
            trend_output_text, trend_image_rel_static_path, pdf_report_rel_static_path, plotly_data = \
                health_analysis.health_trend_analysis(upload_path, base_output_dir, analysis_timestamp_str, time_period, data_type)
            
            if "錯誤" in trend_output_text:
                if user_socket_sid:
                    socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': f'❌ {trend_output_text}'}, room=user_socket_sid)
                return jsonify({'success': False, 'message': trend_output_text}), 500

            trend_url = url_for('static', filename=trend_image_rel_static_path) if trend_image_rel_static_path else None
            pdf_url = url_for('static', filename=pdf_report_rel_static_path) if pdf_report_rel_static_path else None

            response_data = {
                'success': True,
                'message': '趨勢分析完成。',
                'trend_output_text': trend_output_text,
                'trend_url': trend_url,
                'pdf_url': pdf_url,
                'plot_data': plotly_data
            }
            if user_socket_sid:
                socketio.emit('trend_result', {'event_type': 'trend', **response_data}, room=user_socket_sid)
            return jsonify(response_data)

        except Exception as e:
            error_message = f"趨勢分析失敗: {e}"
            import traceback
            traceback.print_exc()
            if user_socket_sid:
                socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': f'❌ {error_message}'}, room=user_socket_sid)
            return jsonify({'success': False, 'message': error_message}), 500
    else:
        if user_socket_sid:
            socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': '❌ 不支援的檔案類型，請上傳 CSV 檔案。'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': '不支援的檔案類型，請上傳 CSV 檔案。'}), 400

@app.route('/analyze_account_trend', methods=['POST'])
@login_required
def analyze_account_trend():
    user_id = current_user.id
    user_socket_sid = user_sid_map.get(user_id)

    time_period = request.form.get('time_period')
    data_type = request.form.get('data_type')

    if data_type == 'blood_pressure':
        csv_file_path = get_user_data_path(user_id, 'data', 'blood_pressure_records.csv')
    elif data_type == 'blood_sugar':
        csv_file_path = get_user_data_path(user_id, 'data', 'blood_sugar_records.csv')
    else:
        if user_socket_sid:
            socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': '❌ 無效的數據類型。'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': '無效的數據類型。'}), 400

    if not os.path.exists(csv_file_path):
        if user_socket_sid:
            socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': '❌ 該數據類型無歷史紀錄可供分析。'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': '該數據類型無歷史紀錄可供分析。'}), 404
    
    clear_user_data_folder(user_id, 'trend')
    clear_user_data_folder(user_id, 'pca')
    clear_user_data_folder(user_id, 'reports')

    analysis_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        base_output_dir = get_user_data_path(user_id)
        trend_output_text, trend_image_rel_static_path, pdf_report_rel_static_path, plotly_data = \
            health_analysis.health_trend_analysis(csv_file_path, base_output_dir, analysis_timestamp_str, time_period, data_type)
        
        if "錯誤" in trend_output_text:
            if user_socket_sid:
                socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': f'❌ {trend_output_text}'}, room=user_socket_sid)
            return jsonify({'success': False, 'message': trend_output_text}), 500

        trend_url = url_for('static', filename=trend_image_rel_static_path) if trend_image_rel_static_path else None
        pdf_url = url_for('static', filename=pdf_report_rel_static_path) if pdf_report_rel_static_path else None

        response_data = {
            'success': True,
            'message': '帳戶數據趨勢分析完成。',
            'trend_output_text': trend_output_text,
            'trend_url': trend_url,
            'pdf_url': pdf_url,
            'plot_data': plotly_data
        }
        if user_socket_sid:
            socketio.emit('trend_result', {'event_type': 'trend', **response_data}, room=user_socket_sid)
        return jsonify(response_data)

    except Exception as e:
        error_message = f"分析帳戶數據趨勢失敗: {e}"
        import traceback
        traceback.print_exc()
        if user_socket_sid:
            socketio.emit('trend_result', {'event_type': 'trend', 'success': False, 'message': f'❌ {error_message}'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': error_message}), 500

@app.route('/generate_and_send_report', methods=['POST'])
@login_required
def generate_and_send_report():
    data = request.get_json()
    recipient_email = data.get('email')
    period = data.get('period')
    data_type = data.get('data_type')
    user_id = current_user.id
    sender_email = "healthllm.team@gmail.com" # Your sending email
    user_socket_sid = user_sid_map.get(user_id)

    if not recipient_email:
        if user_socket_sid:
            socketio.emit('update', {'event_type': 'report', 'success': False, 'message': '❌ 請提供收件人電子郵件。'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': '請提供收件人電子郵件。'}), 400

    if data_type == 'blood_pressure':
        csv_file_path = get_user_data_path(user_id, 'data', 'blood_pressure_records.csv')
        report_subject = f"HealthLLM - 血壓趨勢報告 ({period})"
        report_body = f"您好，<br><br>這是您在 HealthLLM 系統中生成的血壓趨勢報告，分析期間為 {period}。<br><br>請查收附件。<br><br>此致，<br>HealthLLM 團隊"
    elif data_type == 'blood_sugar':
        csv_file_path = get_user_data_path(user_id, 'data', 'blood_sugar_records.csv')
        report_subject = f"HealthLLM - 血糖趨勢報告 ({period})"
        report_body = f"您好，<br><br>這是您在 HealthLLM 系統中生成的血糖趨勢報告，分析期間為 {period}。<br><br>請查收附件。<br><br>此致，<br>HealthLLM 團隊"
    else:
        if user_socket_sid:
            socketio.emit('update', {'event_type': 'report', 'success': False, 'message': '❌ 無效的數據類型。'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': '無效的數據類型。'}), 400

    if not os.path.exists(csv_file_path):
        if user_socket_sid:
            socketio.emit('update', {'event_type': 'report', 'success': False, 'message': '❌ 該數據類型無歷史紀錄可供生成報告。'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': '該數據類型無歷史紀錄可供生成報告。'}), 404

    clear_user_data_folder(user_id, 'reports')
    clear_user_data_folder(user_id, 'trend')
    clear_user_data_folder(user_id, 'pca')

    analysis_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        base_output_dir = get_user_data_path(user_id)
        trend_output_text, trend_image_rel_static_path, pdf_report_rel_static_path, _ = \
            health_analysis.health_trend_analysis(csv_file_path, base_output_dir, analysis_timestamp_str, period, data_type)
        
        if not pdf_report_rel_static_path:
            if user_socket_sid:
                socketio.emit('update', {'event_type': 'report', 'success': False, 'message': '❌ PDF 報告生成失敗，請檢查後端日誌。'}, room=user_socket_sid)
            return jsonify({'success': False, 'message': 'PDF 報告生成失敗，請檢查後端日誌。'}), 500

        abs_pdf_path = os.path.abspath(os.path.join('static', pdf_report_rel_static_path))

        success, message = send_email_with_gmail_api(
            sender_email=sender_email,
            recipient_email=recipient_email,
            subject=report_subject,
            body=report_body,
            attachment_path=abs_pdf_path
        )

        if success:
            if user_socket_sid:
                socketio.emit('update', {'event_type': 'report', 'success': True, 'message': f'🟢 健康報告已成功發送到 {recipient_email}。'}, room=user_socket_sid)
            return jsonify({'success': True, 'message': message})
        else:
            if user_socket_sid:
                socketio.emit('update', {'event_type': 'report', 'success': False, 'message': f'❌ 寄送報告失敗: {message}'}, room=user_socket_sid)
            return jsonify({'success': False, 'message': message}), 500

    except Exception as e:
        print(f"Error generating or sending report: {e}")
        import traceback
        traceback.print_exc()
        if user_socket_sid:
            socketio.emit('update', {'event_type': 'report', 'success': False, 'message': f'❌ 生成或寄送報告失敗: {e}'}, room=user_socket_sid)
        return jsonify({'success': False, 'message': f'生成或寄送報告失敗: {e}'}), 500

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        user_id = current_user.id
        user_sid_map[user_id] = request.sid
        print(f'Client connected: {request.sid} for user {user_id}')
    else:
        print(f'Client connected: {request.sid} (unauthenticated)')

@socketio.on('disconnect')
def handle_disconnect():
    disconnected_sid = request.sid
    user_id_to_remove = None
    for user_id, sid in user_sid_map.items():
        if sid == disconnected_sid:
            user_id_to_remove = user_id
            break
    if user_id_to_remove:
        del user_sid_map[user_id_to_remove]
        print(f'Client disconnected: {disconnected_sid} for user {user_id_to_remove}')
    else:
        print(f'Client disconnected: {disconnected_sid} (unknown user)')

# --- Run the application ---
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Ensure test user data folders exist for initial setup
    test_user_id_for_init = "testuser123" # This should match a user ID from your auth.py or a test user
    os.makedirs(get_user_data_path(test_user_id_for_init, "data"), exist_ok=True)
    
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)


# --- Gmail OAuth2 驗證流程整合 ---
def get_gmail_auth_flow():
    return Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=SCOPES,
        redirect_uri='http://127.0.0.1:5000/'
    )

@app.route('/authorize_gmail')
@login_required
def authorize_gmail():
    """產生 Gmail 授權連結，啟動 OAuth 流程"""
    try:
        flow = get_gmail_auth_flow()
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        with open('flow.pickle', 'wb') as f:
            pickle.dump(flow, f)
        return redirect(auth_url)
    except Exception as e:
        return f"⚠️ 無法建立授權流程: {e}", 500

@app.route('/gmail_callback')
@login_required
def gmail_callback():
    """處理 Google OAuth 授權完成後的回傳"""
    try:
        with open('flow.pickle', 'rb') as f:
            flow = pickle.load(f)
    except Exception as e:
        return f"⚠️ 無法載入授權流程資料: {e}", 500

    try:
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        with open(TOKEN_FILE, 'wb') as token_file:
            pickle.dump(credentials, token_file)
        return '''
        ✅ Gmail 授權成功！<br>
        您現在可以使用郵件發送功能。<br><br>
        <a href="/">回到首頁</a>
        '''
    except Exception as e:
        return f"❌ 授權流程錯誤: {e}", 500
