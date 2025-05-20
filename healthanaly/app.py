import os
import threading
import pandas as pd
import requests
from flask import Flask, render_template, request, send_file, make_response, redirect, url_for, jsonify, Response
from flask_socketio import SocketIO, join_room
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_login import login_required, current_user
from auth import init_auth, get_user_upload_folder, load_user_settings, get_user_by_id
from img_recognition import img_recognition_bp
from health_analysis import (
    health_trend_analysis,
    answer_care_question,
    validate_bp_csv,
    validate_sugar_csv
)
from datetime import datetime
from lib import mdToHtml, clear_user_data_folder

# Load environment variables
load_dotenv()

# Get RAG server URL from environment variables
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL")
if not RAG_SERVER_URL:
    print("Warning: RAG_SERVER_URL not set in environment variables.")

# Allow OAuth over HTTP for development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Flask and SocketIO initialization
app = Flask(__name__)
app.config['TMP_FOLDER'] = 'tmp'
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-secret-key")
app.config['GOOGLE_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")
app.config['OAUTHLIB_INSECURE_TRANSPORT'] = True
if os.getenv('TUNNEL_MODE') == "True":
    app.config['SERVER_NAME'] = os.getenv("SERVER_NAME")

# Initialize authentication
init_auth(app)

# Register Blueprints
app.register_blueprint(img_recognition_bp) # Added blueprint registration

socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        join_room(current_user.id)
        app.logger.info(f"SocketIO: Client {request.sid} joined room {current_user.id} for user {current_user.name}")
    else:
        app.logger.info(f"SocketIO: Anonymous client {request.sid} connected.")

@socketio.on('disconnect')
def handle_disconnect():
    # Rooms are generally cleaned up automatically by Flask-SocketIO when a client disconnects.
    # If explicit leave_room(current_user.id) were needed, current_user might not be reliably available here.
    # Relying on automatic cleanup is standard.
    app.logger.info(f"SocketIO: Client {request.sid} disconnected.")

logLess = False
if logLess:
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print(" * Running on http://127.0.0.1:5000")

# Create upload and temporary folders
os.makedirs(app.config['TMP_FOLDER'], exist_ok=True)

# Context processor to make account_role available to templates
@app.context_processor
def inject_user_role():
    if current_user.is_authenticated:
        user_settings = load_user_settings(current_user.id)
        return dict(account_role=user_settings.get('account_role'))
    return dict(account_role=None)

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
        # For general users, index.html is currently the default.
        # The new dashboard will be on a separate route.
    return render_template('index.html')

# General User Dashboard page
@app.route('/general_dashboard')
@login_required
def general_dashboard():
    user_settings = load_user_settings(current_user.id)
    account_role = user_settings.get('account_role')

    if not account_role:
        # User hasn't completed onboarding, redirect them.
        return redirect(url_for('onboarding'))

    if account_role == 'general':
        # Only general users can access this dashboard.
        # response = make_response(render_template('general_user_dashboard.html'))
        response = make_response(render_template('index_mixed.html'))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    else:
        # If an elderly user or any other role tries to access, redirect to their default page.
        return redirect(url_for('index'))

# API endpoint to get linked accounts
@app.route('/get_linked_accounts', methods=['GET'])
@login_required
def get_linked_accounts_api():
    user_settings = load_user_settings(current_user.id)
    bound_account_ids = user_settings.get('bound_accounts', [])
    
    linked_accounts_details = []
    for account_id in bound_account_ids:
        user = get_user_by_id(account_id) # This function is in auth.py
        if user:
            linked_accounts_details.append({'id': user.id, 'name': user.name})
            
    return jsonify({'accounts': linked_accounts_details})

# API endpoint to get linked account's health data (BP or Sugar) for a specific date
def get_linked_health_data_generic(linked_user_id, data_type_csv_name, target_date_str):
    # Security check: Ensure current_user is linked to linked_user_id
    current_user_settings = load_user_settings(current_user.id)
    if (linked_user_id not in current_user_settings.get('bound_accounts', [])) and (current_user.id != linked_user_id):
        return jsonify({'error': 'Unauthorized access to linked account data.'}), 403

    if not target_date_str:
        return jsonify({'error': 'Missing date parameter.'}), 400

    linked_user_folder = get_user_upload_folder(linked_user_id)
    csv_path = os.path.join(linked_user_folder, data_type_csv_name)
    
    record_for_date = None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Ensure 'Date' column is string type for comparison, and handle potential NaN before astype(str)
            df['Date'] = df['Date'].fillna('').astype(str)
            
            # Filter for the target date
            # Use .loc to avoid SettingWithCopyWarning if we were to modify, though here we are just reading
            date_match_df = df.loc[df['Date'] == target_date_str]
            
            if not date_match_df.empty:
                # Take the last entry for the date if multiple exist (though ideally there should be one)
                # Convert that row to a dictionary. Replace NaN with empty strings.
                record_for_date = date_match_df.iloc[-1].fillna('').astype(str).to_dict()
                
                # Frontend expects keys like 'morning_systolic', backend CSV has 'Morning_Systolic'
                # We need to map CSV column names to frontend expected names if they differ.
                # For now, let's assume the frontend will handle the CSV column names directly if they are consistent,
                # or the populateBpForm/populateSugarForm will map them.
                # The current populateBpForm expects lowercase_with_underscore.
                # Let's transform keys here to match frontend expectations.
                
                if record_for_date:
                    transformed_record = {}
                    for key, value in record_for_date.items():
                        # Convert CamelCase from CSV to snake_case for frontend
                        new_key = key.lower()
                        transformed_record[new_key] = value
                    record_for_date = transformed_record

        except pd.errors.EmptyDataError:
            app.logger.info(f"{data_type_csv_name} for user {linked_user_id} is empty.")
            # record_for_date remains None
        except Exception as e:
            app.logger.error(f"Error reading {data_type_csv_name} for user {linked_user_id} on date {target_date_str}: {e}")
            return jsonify({'error': f'Error reading {data_type_csv_name}.'}), 500
            
    return jsonify({'record': record_for_date})

@app.route('/get_linked_bp_data', methods=['GET'])
@login_required
def get_linked_bp_data_api():
    linked_user_id = request.args.get('user_id')
    target_date = request.args.get('date') # Get the date from query parameters
    if not linked_user_id:
        return jsonify({'error': 'Missing user_id parameter.'}), 400
    if not target_date:
        return jsonify({'error': 'Missing date parameter.'}), 400
    return get_linked_health_data_generic(linked_user_id, 'blood_pressure.csv', target_date)

@app.route('/get_linked_sugar_data', methods=['GET'])
@login_required
def get_linked_sugar_data_api():
    linked_user_id = request.args.get('user_id')
    target_date = request.args.get('date') # Get the date from query parameters
    if not linked_user_id:
        return jsonify({'error': 'Missing user_id parameter.'}), 400
    if not target_date:
        return jsonify({'error': 'Missing date parameter.'}), 400
    return get_linked_health_data_generic(linked_user_id, 'blood_sugar.csv', target_date)

# API endpoint to update linked account's health data
def update_linked_health_data_generic(linked_user_id, records_data, data_type):
    # Security check
    current_user_settings = load_user_settings(current_user.id)
    if (linked_user_id not in current_user_settings.get('bound_accounts', [])) and (current_user.id != linked_user_id):
        return jsonify({'success': False, 'message': '未授權更新此帳戶的資料。'}), 403

    linked_user_folder = get_user_upload_folder(linked_user_id)

    validated_records = []
    for record_dict in records_data:
        # Ensure 'date' is present
        if 'date' not in record_dict or not record_dict['date']:
            return jsonify({'success': False, 'message': '記錄中缺少日期欄位。'}), 400
        
        # Replace None with '無' for validation, as '無' is the expected string for empty in validation
        processed_record_for_validation = {k: (v if v is not None else '無') for k, v in record_dict.items()}

        validation_error = validate_health_data(processed_record_for_validation, data_type)
        if validation_error:
            return jsonify({'success': False, 'message': f"資料驗證失敗 (日期 {record_dict['date']}): {validation_error}"}), 400
        
        validated_records.append(record_dict)

    try:
        # Use save_health_data_background for each record with overwrite=True
        for record in validated_records:
            threading.Thread(target=save_health_data_background, args=(record, data_type, linked_user_folder, linked_user_id, True)).start()

        # Emit update to the current user who performed the action
        socketio.emit('update', {'message': f'🟢 {data_type.replace("_", " ")} 資料已成功更新。', 'event_type': 'summary', 'target': f'linked_{data_type}'}, room=current_user.id)
        return jsonify({'success': True, 'message': f'{data_type.replace("_", " ")} 資料已成功更新。'})
    except Exception as e:
        app.logger.error(f"Error saving linked {data_type} data for user {linked_user_id}: {e}")
        socketio.emit('update', {'message': f'❌ 更新連結帳戶 {data_type.replace("_", " ")} 資料時發生錯誤: {str(e)}', 'event_type': 'summary', 'target': f'linked_{data_type}'}, room=current_user.id)
        return jsonify({'success': False, 'message': f'儲存失敗: {str(e)}'}), 500

@app.route('/update_linked_bp_data', methods=['POST'])
@login_required
def update_linked_bp_data_api():
    data = request.get_json()
    linked_user_id = data.get('user_id')
    records = data.get('records')
    if not linked_user_id or records is None: # records can be an empty list
        return jsonify({'success': False, 'message': '缺少 user_id 或 records 資料。'}), 400
    return update_linked_health_data_generic(linked_user_id, records, 'blood_pressure')

@app.route('/update_linked_sugar_data', methods=['POST'])
@login_required
def update_linked_sugar_data_api():
    data = request.get_json()
    linked_user_id = data.get('user_id')
    records = data.get('records')
    if not linked_user_id or records is None:
        return jsonify({'success': False, 'message': '缺少 user_id 或 records 資料。'}), 400
    return update_linked_health_data_generic(linked_user_id, records, 'blood_sugar')

# Validate input values
def validate_health_data(data, data_type):
    if data_type == 'blood_pressure':
        fields = {
            'morning_systolic': (50, 250),
            'morning_diastolic': (30, 150),
            'morning_pulse': (30, 200),
            'noon_systolic': (50, 250),    # Added noon
            'noon_diastolic': (30, 150),   # Added noon
            'noon_pulse': (30, 200),       # Added noon
            'evening_systolic': (50, 250),
            'evening_diastolic': (30, 150),
            'evening_pulse': (30, 200)
        }
    elif data_type == 'blood_sugar':
        fields = {
            'morning_fasting': (50, 300),
            'morning_postprandial': (70, 400),
            'noon_fasting': (50, 300),       # Added noon
            'noon_postprandial': (70, 400),  # Added noon
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
def save_health_data_background(data_dict, data_type, user_folder, user_id, overwrite=False):
    try:
        # Convert form data to DataFrame
        data = {'Date': [data_dict['date']]}
        if data_type == 'blood_pressure':
            data.update({
                'Morning_Systolic': [data_dict.get('morning_systolic', '無')],
                'Morning_Diastolic': [data_dict.get('morning_diastolic', '無')],
                'Morning_Pulse': [data_dict.get('morning_pulse', '無')],
                'Noon_Systolic': [data_dict.get('noon_systolic', '無')],      # Added noon
                'Noon_Diastolic': [data_dict.get('noon_diastolic', '無')],    # Added noon
                'Noon_Pulse': [data_dict.get('noon_pulse', '無')],          # Added noon
                'Evening_Systolic': [data_dict.get('evening_systolic', '無')],
                'Evening_Diastolic': [data_dict.get('evening_diastolic', '無')],
                'Evening_Pulse': [data_dict.get('evening_pulse', '無')]
            })
        elif data_type == 'blood_sugar':
            data.update({
                'Morning_Fasting': [data_dict.get('morning_fasting', '無')],
                'Morning_Postprandial': [data_dict.get('morning_postprandial', '無')],
                'Noon_Fasting': [data_dict.get('noon_fasting', '無')],          # Added noon
                'Noon_Postprandial': [data_dict.get('noon_postprandial', '無')],# Added noon
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
            # Ensure Date column is string for comparison
            existing_df['Date'] = existing_df['Date'].astype(str)
            
            if data_dict['date'] in existing_df['Date'].values:
                if not overwrite:
                    # Check if new data is actually different from existing data for that date
                    existing_row = existing_df[existing_df['Date'] == data_dict['date']].iloc[-1]
                    prompt_for_overwrite = False
                    fields_to_check = []
                    if data_type == 'blood_pressure':
                        fields_to_check = [
                            ('morning_systolic', 'Morning_Systolic'), ('morning_diastolic', 'Morning_Diastolic'), ('morning_pulse', 'Morning_Pulse'),
                            ('noon_systolic', 'Noon_Systolic'), ('noon_diastolic', 'Noon_Diastolic'), ('noon_pulse', 'Noon_Pulse'), # Added noon
                            ('evening_systolic', 'Evening_Systolic'), ('evening_diastolic', 'Evening_Diastolic'), ('evening_pulse', 'Evening_Pulse')
                        ]
                    elif data_type == 'blood_sugar':
                        fields_to_check = [
                            ('morning_fasting', 'Morning_Fasting'), ('morning_postprandial', 'Morning_Postprandial'),
                            ('noon_fasting', 'Noon_Fasting'), ('noon_postprandial', 'Noon_Postprandial'), # Added noon
                            ('evening_fasting', 'Evening_Fasting'), ('evening_postprandial', 'Evening_Postprandial')
                        ]
                    
                    for form_key, csv_key in fields_to_check:
                        new_value = str(data_dict.get(form_key, '無')).strip()
                        existing_value = str(existing_row.get(csv_key, '無')).strip()
                        if new_value != '無' and new_value != existing_value:
                            prompt_for_overwrite = True
                            break
                    
                    if prompt_for_overwrite:
                        socketio.emit('confirm_overwrite', {
                            'message': f'🟡 {data_type.replace("_", " ")} 當日資料已存在且與提交內容不同，是否覆蓋？',
                            'data': data_dict,
                            'data_type': data_type,
                            'user_folder': user_folder, 
                            'event_type': 'summary'
                        }, room=user_id)
                        return # Wait for user confirmation via /overwrite_health_data
                
                # If overwrite is True, or if not prompting for overwrite (e.g. new data matches existing or only fills '無')
                existing_df = existing_df[existing_df['Date'] != data_dict['date']] # Remove old row(s) for this date
                new_df = pd.concat([existing_df, df], ignore_index=True)
            else: # Date not in existing_df
                new_df = pd.concat([existing_df, df], ignore_index=True)
        else: # CSV file does not exist
            new_df = df
        
        new_df.sort_values(by='Date', inplace=True) # Keep CSV sorted by date
        new_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        socketio.emit('update', {'message': f'🟢 {data_type.replace("_", " ")}紀錄儲存成功', 'event_type': 'summary'}, room=user_id)
    except Exception as e:
        app.logger.error(f"Error in save_health_data_background for {data_type}, user {user_id} (folder {user_folder}): {str(e)}")
        socketio.emit('update', {'message': f"❌ {data_type.replace('_', ' ')}紀錄儲存錯誤: {str(e)}", 'event_type': 'summary'}, room=user_id)

# Background task for trend analysis
def trend_background_task(file_path, user_id):
    try:
        df = pd.read_csv(file_path)
        df.fillna("無", inplace=True)

        # Determine data_type based on CSV validation
        data_type = 'blood_pressure' if validate_bp_csv(df) else 'blood_sugar' if validate_sugar_csv(df) else None
        if not data_type:
            socketio.emit('update', {'message': '❌ CSV 檔案格式不符合血壓或血糖分析要求', 'event_type': 'trend'}, room=user_id)
            return

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        clear_user_data_folder(user_id, "trend")
        result = health_trend_analysis(file_path, user_id, timestamp)
        socketio.emit('update', {'message': '🟢 趨勢分析完成', 'event_type': 'trend'}, room=user_id)
        socketio.emit('trend_result', {
            'trend_output': result,
            'trend_url': f'/download_trend/{user_id}/{data_type}/{timestamp}',
            'event_type': 'trend'
        }, room=user_id)
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 趨勢分析錯誤: {str(e)}", 'event_type': 'trend'}, room=user_id)

# Save health data to CSV
@app.route('/save_health_data', methods=['POST'])
@login_required
def save_health_data():
    try:
        form_date = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        user_folder = get_user_upload_folder(current_user.id)
        
        # Updated field lists to include noon
        bp_data_fields = ['morning_systolic', 'morning_diastolic', 'morning_pulse', 
                          'noon_systolic', 'noon_diastolic', 'noon_pulse',
                          'evening_systolic', 'evening_diastolic', 'evening_pulse']
        sugar_data_fields = ['morning_fasting', 'morning_postprandial', 
                             'noon_fasting', 'noon_postprandial',
                             'evening_fasting', 'evening_postprandial']

        # Check for actual BP data (morning, noon, or evening)
        actual_bp_data_present = any(request.form.get(field) and request.form.get(field) != '無' for field in bp_data_fields)
        
        # Check for actual Sugar data (morning, noon, or evening)
        actual_sugar_data_present = any(request.form.get(field) and request.form.get(field) != '無' for field in sugar_data_fields)

        bp_processed_successfully = False
        sugar_processed_successfully = False
        at_least_one_type_had_data = False


        if actual_bp_data_present:
            at_least_one_type_had_data = True
            bp_payload = {'date': form_date}
            for field in bp_data_fields: # Iterate over all BP fields including noon
                bp_payload[field] = request.form.get(field, '無')
            
            validation_error_bp = validate_health_data(bp_payload, 'blood_pressure')
            if validation_error_bp:
                socketio.emit('update', {'message': f"❌ 血壓輸入錯誤: {validation_error_bp}", 'event_type': 'summary'}, room=current_user.id)
            else:
                socketio.emit('update', {'message': '🟢 血壓資料上傳成功，開始儲存...', 'event_type': 'summary'}, room=current_user.id)
                threading.Thread(target=save_health_data_background, args=(bp_payload, 'blood_pressure', user_folder, current_user.id)).start()
                bp_processed_successfully = True
        
        if actual_sugar_data_present:
            at_least_one_type_had_data = True
            sugar_payload = {'date': form_date}
            for field in sugar_data_fields: # Iterate over all Sugar fields including noon
                sugar_payload[field] = request.form.get(field, '無')

            validation_error_sugar = validate_health_data(sugar_payload, 'blood_sugar')
            if validation_error_sugar:
                socketio.emit('update', {'message': f"❌ 血糖輸入錯誤: {validation_error_sugar}", 'event_type': 'summary'}, room=current_user.id)
            else:
                socketio.emit('update', {'message': '🟢 血糖資料上傳成功，開始儲存...', 'event_type': 'summary'}, room=current_user.id)
                threading.Thread(target=save_health_data_background, args=(sugar_payload, 'blood_sugar', user_folder, current_user.id)).start()
                sugar_processed_successfully = True

        if not at_least_one_type_had_data:
            socketio.emit('update', {'message': '⚠️ 未輸入任何健康數據。', 'event_type': 'summary'}, room=current_user.id)
            return '未輸入任何健康數據。', 400
        
        # If at least one type had data, but neither was processed successfully due to validation errors
        if at_least_one_type_had_data and not bp_processed_successfully and not sugar_processed_successfully:
             # Errors already emitted by validation checks, so just return 400
            return '輸入數據驗證失敗。', 400

        return '資料已提交處理。', 200
    except Exception as e:
        app.logger.error(f"Error in save_health_data: {str(e)}")
        socketio.emit('update', {'message': f"❌ 伺服器內部錯誤: {str(e)}", 'event_type': 'summary'}, room=current_user.id)
        return f'伺服器內部錯誤: {str(e)}', 500

# API endpoint to get today's health data
@app.route('/api/get_today_health_data', methods=['GET'])
@login_required
def get_today_health_data_api():
    user_folder = get_user_upload_folder(current_user.id)
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    
    data_to_return = {}

    # --- Blood Pressure ---
    bp_csv_path = os.path.join(user_folder, 'blood_pressure.csv')
    if os.path.exists(bp_csv_path):
        try:
            bp_df = pd.read_csv(bp_csv_path)
            bp_df['Date'] = bp_df['Date'].astype(str)
            today_bp_row_df = bp_df[bp_df['Date'] == today_date_str]
            if not today_bp_row_df.empty:
                row = today_bp_row_df.iloc[-1] # Get the last entry for the day if multiple
                data_to_return['morning_systolic'] = str(row.get('Morning_Systolic', ''))
                data_to_return['morning_diastolic'] = str(row.get('Morning_Diastolic', ''))
                data_to_return['morning_pulse'] = str(row.get('Morning_Pulse', ''))
                data_to_return['noon_systolic'] = str(row.get('Noon_Systolic', ''))      # Added noon
                data_to_return['noon_diastolic'] = str(row.get('Noon_Diastolic', ''))    # Added noon
                data_to_return['noon_pulse'] = str(row.get('Noon_Pulse', ''))          # Added noon
                data_to_return['evening_systolic'] = str(row.get('Evening_Systolic', ''))
                data_to_return['evening_diastolic'] = str(row.get('Evening_Diastolic', ''))
                data_to_return['evening_pulse'] = str(row.get('Evening_Pulse', ''))
        except Exception as e:
            app.logger.error(f"Error reading blood_pressure.csv for user {current_user.id} on {today_date_str}: {e}")

    # --- Blood Sugar ---
    bs_csv_path = os.path.join(user_folder, 'blood_sugar.csv')
    if os.path.exists(bs_csv_path):
        try:
            bs_df = pd.read_csv(bs_csv_path)
            bs_df['Date'] = bs_df['Date'].astype(str)
            today_bs_row_df = bs_df[bs_df['Date'] == today_date_str]
            if not today_bs_row_df.empty:
                row = today_bs_row_df.iloc[-1] # Get the last entry for the day
                data_to_return['morning_fasting'] = str(row.get('Morning_Fasting', ''))
                data_to_return['morning_postprandial'] = str(row.get('Morning_Postprandial', ''))
                data_to_return['noon_fasting'] = str(row.get('Noon_Fasting', ''))          # Added noon
                data_to_return['noon_postprandial'] = str(row.get('Noon_Postprandial', ''))# Added noon
                data_to_return['evening_fasting'] = str(row.get('Evening_Fasting', ''))
                data_to_return['evening_postprandial'] = str(row.get('Evening_Postprandial', ''))
        except Exception as e:
            app.logger.error(f"Error reading blood_sugar.csv for user {current_user.id} on {today_date_str}: {e}")
            
    # Replace '無' and pandas NaN with empty strings for form inputs
    for key, value in data_to_return.items():
        if pd.isna(value) or str(value).strip() == '無':
            data_to_return[key] = ''
            
    return jsonify(data_to_return)

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
        
        thread = threading.Thread(target=save_health_data_background, args=(data_dict, data_type, user_folder, current_user.id, True))
        thread.start()
        return '覆蓋資料已處理。', 200
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 覆蓋資料錯誤: {str(e)}", 'event_type': 'summary'}, room=current_user.id)
        return f'覆蓋資料錯誤: {str(e)}', 500

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

    socketio.emit('update', {'message': '🟢 檔案上傳成功，開始趨勢分析...', 'event_type': 'trend'}, room=current_user.id)
    thread = threading.Thread(target=trend_background_task, args=(file_path, current_user.id))
    thread.start()
    return '檔案已上傳並開始處理。', 200

# Upload CSV for trend analysis for a LINKED account
@app.route('/upload_trend_linked', methods=['POST'])
@login_required
def upload_trend_linked():
    file = request.files.get('file')
    linked_user_id = request.form.get('user_id')

    if not file or file.filename == '':
        return jsonify({'success': False, 'message': '請選擇檔案'}), 400
    if not linked_user_id:
        return jsonify({'success': False, 'message': '缺少連結帳戶 user_id'}), 400

    # Security check: Ensure current_user is linked to linked_user_id
    current_user_settings = load_user_settings(current_user.id)
    if (linked_user_id not in current_user_settings.get('bound_accounts', [])) and (current_user.id != linked_user_id):
        return jsonify({'success': False, 'message': '未授權分析此帳戶的資料。'}), 403

    filename = secure_filename(file.filename)
    # Save to the linked user's folder
    linked_user_folder = get_user_upload_folder(linked_user_id)
    file_path = os.path.join(linked_user_folder, filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        app.logger.error(f"Error saving uploaded trend file for linked user {linked_user_id}: {e}")
        socketio.emit('update', {'message': f"❌ 上傳趨勢分析檔案失敗: {str(e)}", 'event_type': 'trend', 'target': 'trend'}, room=current_user.id)
        return jsonify({'success': False, 'message': f'檔案儲存失敗: {str(e)}'}), 500

    socketio.emit('update', {'message': '🟢 連結帳戶檔案上傳成功，開始趨勢分析...', 'event_type': 'trend', 'target': 'trend'}, room=current_user.id)
    # Use linked_user_id for the background task, so trend results go to the linked user
    thread = threading.Thread(target=trend_background_task, args=(file_path, linked_user_id))
    thread.start()
    return jsonify({'success': True, 'message': '連結帳戶檔案已上傳並開始處理。'}), 200


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
        socketio.emit('question_result', {'answer': answer_html, 'event_type': 'question'}, room=current_user.id)
        return '問題已處理', 200
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 問題回答錯誤: {str(e)}", 'event_type': 'question'}, room=current_user.id)
    return '問題處理錯誤', 500

# Answer caregiver questions for a LINKED account
@app.route('/ask_question_linked', methods=['POST'])
@login_required
def ask_question_linked():
    question = request.form.get('question', '').strip()
    linked_user_id = request.form.get('user_id')

    if not question:
        return jsonify({'success': False, 'message': '請輸入問題'}), 400
    if not linked_user_id:
        return jsonify({'success': False, 'message': '缺少連結帳戶 user_id'}), 400

    # Security check: Ensure current_user is linked to linked_user_id
    current_user_settings = load_user_settings(current_user.id)
    if (linked_user_id not in current_user_settings.get('bound_accounts', [])) and (current_user.id != linked_user_id):
        return jsonify({'success': False, 'message': '未授權查詢此帳戶的問題。'}), 403
    
    # For now, answer_care_question is generic. If it needs linked_user_id context,
    # it would need to be passed, e.g., answer_care_question(question, linked_user_id)
    try:
        answer = answer_care_question(question) # Potentially pass linked_user_id if needed by the function
        answer_html = mdToHtml(answer)
        # The 'target' for socketio event might need adjustment if specific UI elements for linked Q&A exist
        socketio.emit('question_result', {'answer': answer_html, 'event_type': 'question', 'target': 'question'}, room=current_user.id)
        return jsonify({'success': True, 'message': '連結帳戶問題已處理'}), 200
    except Exception as e:
        app.logger.error(f"Error answering linked question for user {linked_user_id}: {e}")
        socketio.emit('update', {'message': f"❌ 連結帳戶問題回答錯誤: {str(e)}", 'event_type': 'question', 'target': 'question'}, room=current_user.id)
        return jsonify({'success': False, 'message': '問題處理錯誤'}), 500

# RAG Chat page
@app.route('/rag_chat')
@login_required
def rag_chat():
    response = make_response(render_template('rag_chat.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Handle text questions for RAG
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
        # Send question to RAG server chat endpoint
        chat_response = requests.post(f"{RAG_SERVER_URL}/submit", json={'question': question})
        chat_response.raise_for_status() # Raise an exception for bad status codes
        rag_answer = mdToHtml(chat_response.json().get('answer', '無法取得回答'))

        audio_url = None
        if voice_mode:
            # Send answer to RAG server text-to-speech endpoint
            tts_response = requests.post(f"{RAG_SERVER_URL}/record", json={'text': rag_answer})
            tts_response.raise_for_status()
            audio_url = tts_response.json().get('audio_url')

        return jsonify({'answer': rag_answer, 'audio': audio_url})

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error communicating with RAG server: {e}")
        return jsonify({'error': f'與 RAG 伺服器通訊錯誤: {e}'}), 500
    except Exception as e:
        app.logger.error(f"Error processing RAG submit: {e}")
        return jsonify({'error': f'處理 RAG 提交錯誤: {e}'}), 500

# Handle voice input for RAG
@app.route('/rag_record', methods=['POST'])
@login_required
def rag_record():
    audio_file = request.files.get('audio')
    mode = request.form.get('mode') # 'transcribe' or 'voice'

    if not audio_file:
        return jsonify({'error': '未收到音訊檔案'}), 400

    if not RAG_SERVER_URL:
        return jsonify({'error': 'RAG 伺服器地址未設定'}), 500

    try:
        # Send audio to RAG server speech-to-text endpoint
        files = {
            'audio': (audio_file.filename, audio_file.stream, audio_file.mimetype),
            'mode': (None, mode) # Include mode directly in files
        }
        stt_response = requests.post(f"{RAG_SERVER_URL}/record", files=files)
        stt_response.raise_for_status()
        # The RAG server's /record endpoint handles the full voice process when mode='voice'
        rag_response_json = stt_response.json()
        transcription = rag_response_json.get('transcription', '無法轉錄')
        rag_answer = mdToHtml(rag_response_json.get('answer', None)) # Answer is only present in voice mode
        audio_url = rag_response_json.get('audio', None) # Audio URL is only present in voice mode

        return jsonify({'transcription': transcription, 'answer': rag_answer, 'audio': audio_url})

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
        # Return a default error audio file
        return send_file('healthanaly/static/error_audio.mp3', mimetype='audio/mpeg')

    try:
        rag_audio_url = f"{RAG_SERVER_URL}/audio/{filename}"
        response = requests.get(rag_audio_url, stream=True)
        response.raise_for_status()

        # Determine mimetype, default to MP3
        content_type = response.headers.get('Content-Type', 'audio/mpeg')
        
        # Return the audio file directly to the client
        return Response(
            response.iter_content(chunk_size=1024),
            status=response.status_code,
            mimetype=content_type,
            headers={'Content-Disposition': f'inline; filename={filename}'}
        )
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching audio from RAG server: {e}")
        # Return a default error audio file
        return send_file('healthanaly/static/error_audio.mp3', mimetype='audio/mpeg')


# Start server
if __name__ == '__main__':
    socketio.run(app, debug=True)
