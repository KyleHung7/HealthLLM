import os
from flask import redirect, url_for, session, Blueprint, render_template, make_response, request
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from authlib.integrations.flask_client import OAuth
import json
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
login_manager = LoginManager()
oauth = OAuth()

# 使用者模型
class User(UserMixin):
    def __init__(self, id, email, name, role='general'):
        self.id = id
        self.email = email
        self.name = name
        self.role = role

# 使用者緩存 (全局字典，用於 Flask-Login 載入用戶)
users = {}

def init_auth(app):
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    # 設定 Google OAuth
    oauth.init_app(app)
    oauth.register(
        name='google',
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )
    
    app.register_blueprint(auth_bp)

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# 登入路由
@auth_bp.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    nonce = os.urandom(16).hex()
    session['nonce'] = nonce
    redirect_uri = url_for('auth.callback', _external=True)
    if os.getenv('TUNNEL_MODE') == "True":
        server_name = os.getenv("SERVER_NAME")
        if server_name:
            redirect_uri = url_for('auth.callback', _external=True, _scheme='https', _external_host=server_name)
            print("Custom server name redirect: ", redirect_uri)
    return oauth.google.authorize_redirect(redirect_uri, nonce=nonce)

# OAuth 回調路由
@auth_bp.route('/callback')
def callback():
    try:
        token = oauth.google.authorize_access_token()
        if not token:
            return 'Failed to get token', 400
            
        nonce = session.pop('nonce', None)
        if not nonce:
            return 'Invalid session or nonce missing', 400
            
        user_info = oauth.google.parse_id_token(token, nonce=nonce)
        if not user_info:
            return 'Failed to get user info', 400

        user_id = user_info['sub']
        user_email = user_info['email']
        user_name = user_info.get('name', user_info['email'])

        if user_id not in users:
            user = User(
                id=user_id,
                email=user_email,
                name=user_name,
                role='general'
            )
            users[user.id] = user
        else:
            user = users[user_id]
            user.email = user_email
            user.name = user_name

        login_user(user)
        
        user_folder = os.path.join('static', 'user_data', secure_filename(user.id))
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            os.makedirs(os.path.join(user_folder, 'data'), exist_ok=True)
            os.makedirs(os.path.join(user_folder, 'trend'), exist_ok=True)
            os.makedirs(os.path.join(user_folder, 'reports'), exist_ok=True)
            
        user_settings = load_user_settings(user.id)
        user_settings['email'] = user_email
        user_settings['name'] = user_name
        user_settings['role'] = user.role
        save_user_settings(user.id, user_settings)
        
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Auth callback error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Authentication error: {str(e)}", 500

# 登出路由
@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

# 個人資料頁面
@auth_bp.route('/profile')
@login_required
def profile():
    user_settings = load_user_settings(current_user.id)
    ai_enabled = user_settings.get('ai_enabled', True)
    email_report_enabled = user_settings.get('email_report_enabled', True)
    account_role = user_settings.get('account_role', 'general')
    response = make_response(render_template('profile.html', ai_enabled=ai_enabled, email_report_enabled=email_report_enabled, account_role=account_role))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@auth_bp.route('/save_ai_setting', methods=['POST'])
@login_required
def save_ai_setting():
    ai_enabled = request.json.get('ai_enabled')
    user_settings = load_user_settings(current_user.id)
    user_settings['ai_enabled'] = ai_enabled
    save_user_settings(current_user.id, user_settings)
    return 'AI setting saved successfully', 200

@auth_bp.route('/save_email_report_setting', methods=['POST'])
@login_required
def save_email_report_setting():
    email_report_enabled = request.json.get('email_report_enabled')
    user_settings = load_user_settings(current_user.id)
    user_settings['email_report_enabled'] = email_report_enabled
    save_user_settings(current_user.id, user_settings)
    return 'Email report setting saved successfully', 200

@auth_bp.route('/save_account_role', methods=['POST'])
@login_required
def save_account_role():
    account_role = request.json.get('account_role')
    if account_role not in ['elderly', 'general']:
        return 'Invalid account role', 400
        
    user_settings = load_user_settings(current_user.id)
    user_settings['account_role'] = account_role
    save_user_settings(current_user.id, user_settings)
    if current_user.id in users:
        users[current_user.id].role = account_role
    return 'Account role saved successfully', 200

@auth_bp.route('/binding/add', methods=['GET', 'POST'])
@login_required
def account_binding():
    user_settings = load_user_settings(current_user.id)
    account_role = user_settings.get('account_role')

    if account_role != 'general':
        return '您沒有權限存取此頁面', 403

    if request.method == 'POST':
        email = request.form.get('email').lower()
        user = get_user_by_email(email)
        if user:
            if send_binding_request(email, current_user.email):
                store_binding_request(email, current_user.id, current_user.email)
                return redirect(url_for('auth.binding'))
            else:
                return 'Failed to send binding request', 500
        else:
            return 'User with this email does not exist', 400

    return render_template('add_binding.html')

@auth_bp.route('/binding')
@login_required
def binding():
    user_id = current_user.id
    received_requests = [req for req in binding_requests if req['email'] == current_user.email]
    sent_requests = [req for req in binding_requests if req['request_user_id'] == user_id]
    user_settings = load_user_settings(current_user.id)
    bound_accounts = user_settings.get('bound_accounts', [])
    bound_account_emails = [get_user_by_id(account_id).email for account_id in bound_accounts if get_user_by_id(account_id)]
    return render_template('binding.html', received_requests=received_requests, sent_requests=sent_requests, bound_accounts=bound_account_emails)

@auth_bp.route('/binding/accept/<request_user_id>')
@login_required
def accept_binding(request_user_id):
    global binding_requests
    binding_requests_list = [req for req in binding_requests if req['email'] == current_user.email and req['request_user_id'] == request_user_id]
    if binding_requests_list:
        binding_request = binding_requests_list[0]
        binding_requests.remove(binding_request)

        user_settings = load_user_settings(current_user.id)
        bound_accounts = user_settings.get('bound_accounts', [])
        if request_user_id not in bound_accounts:
            bound_accounts.append(request_user_id)
            user_settings['bound_accounts'] = bound_accounts
            save_user_settings(current_user.id, user_settings)

        request_user = get_user_by_email(binding_request['request_user_email'])
        if request_user:
            request_user_settings = load_user_settings(request_user.id)
            request_user_bound_accounts = request_user_settings.get('bound_accounts', [])
            if current_user.id not in request_user_bound_accounts:
                request_user_bound_accounts.append(current_user.id)
                request_user_settings['bound_accounts'] = request_user_bound_accounts
                save_user_settings(request_user.id, request_user_settings)

    return redirect(url_for('auth.binding'))

@auth_bp.route('/binding/reject/<request_user_id>')
@login_required
def reject_binding(request_user_id):
    global binding_requests
    binding_requests_list = [req for req in binding_requests if req['email'] == current_user.email and req['request_user_id'] == request_user_id]
    if binding_requests_list:
        binding_request = binding_requests_list[0]
        binding_requests.remove(binding_request)
    return redirect(url_for('auth.binding'))

@auth_bp.route('/binding/withdraw/<email>')
@login_required
def withdraw_binding(email):
    global binding_requests
    binding_requests_list = [req for req in binding_requests if req['email'] == email and req['request_user_id'] == current_user.id]
    if binding_requests_list:
        binding_request = binding_requests_list[0]
        binding_requests.remove(binding_request)
    return redirect(url_for('auth.binding'))

def get_user_upload_folder(user_id):
    user_folder = os.path.join('static', 'user_data', secure_filename(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def load_user_settings(user_id):
    user_folder = os.path.join('static', 'user_data', secure_filename(user_id))
    settings_file = os.path.join(user_folder, 'settings.json')
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            return settings
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def save_user_settings(user_id, settings):
    user_folder = os.path.join('static', 'user_data', secure_filename(user_id))
    settings_file = os.path.join(user_folder, 'settings.json')
    os.makedirs(user_folder, exist_ok=True)
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)

binding_requests = []

def send_binding_request(email, current_email):
    print(f"Binding request sent to {email} from user {current_email}")
    return True

def store_binding_request(email, user_id, current_email):
    binding_requests.append({'email': email, 'request_user_id': user_id, 'request_user_email': current_email})
    print(f"Binding request stored for {email} from user {current_email}")
    return True

def get_user_by_id(user_id):
    user = users.get(user_id)
    if user:
        return user
    
    user_settings = load_user_settings(user_id)
    if user_settings:
        user = User(
            id=user_id,
            email=user_settings.get('email'),
            name=user_settings.get('name', user_settings.get('email')),
            role=user_settings.get('role', 'general')
        )
        users[user_id] = user
        return user
    return None

def get_user_by_email(email):
    for user_id, user_obj in users.items():
        if user_obj.email == email:
            return user_obj
            
    users_base_dir = os.path.join('static', 'user_data')
    if os.path.exists(users_base_dir):
        for user_id_folder in os.listdir(users_base_dir):
            user_folder_path = os.path.join(users_base_dir, user_id_folder)
            if os.path.isdir(user_folder_path):
                settings_file = os.path.join(user_folder_path, 'settings.json')
                if os.path.exists(settings_file):
                    try:
                        with open(settings_file, 'r', encoding='utf-8') as f:
                            settings = json.load(f)
                            if settings.get('email') == email:
                                user = User(
                                    id=user_id_folder,
                                    email=settings.get('email'),
                                    name=settings.get('name', settings.get('email')),
                                    role=settings.get('role', 'general')
                                )
                                users[user.id] = user
                                return user
                    except (FileNotFoundError, json.JSONDecodeError):
                        continue
    return None

@auth_bp.route('/binding/remove/<email>')
@login_required
def remove_binding(email):
    user_settings = load_user_settings(current_user.id)
    if user_settings.get('account_role') == 'elderly':
        return "Elderly accounts cannot remove bound accounts.", 403

    bound_accounts = user_settings.get('bound_accounts', [])

    user_to_remove = get_user_by_email(email)
    if user_to_remove:
        account_id_to_remove = user_to_remove.id
        if account_id_to_remove in bound_accounts:
            bound_accounts.remove(account_id_to_remove)
            user_settings['bound_accounts'] = bound_accounts
            save_user_settings(current_user.id, user_settings)

            other_user_settings = load_user_settings(account_id_to_remove)
            other_bound_accounts = other_user_settings.get('bound_accounts', [])
            if current_user.id in other_bound_accounts:
                other_bound_accounts.remove(current_user.id)
                other_user_settings['bound_accounts'] = other_bound_accounts
                save_user_settings(account_id_to_remove, other_user_settings)

    return redirect(url_for('auth.binding'))