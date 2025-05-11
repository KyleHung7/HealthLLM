import os
from flask import redirect, url_for, session, Blueprint, render_template, make_response, request
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from authlib.integrations.flask_client import OAuth
import json
from constants.default_settings import default
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
login_manager = LoginManager()
oauth = OAuth()

# 使用者模型
class User(UserMixin):
    def __init__(self, id, email, name):
        self.id = id
        self.email = email
        self.name = name

# 使用者緩存
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
    # Generate and store nonce in session
    nonce = os.urandom(16).hex()
    session['nonce'] = nonce
    redirect_uri = url_for('auth.callback', _external=True)
    if os.getenv('TUNNEL_MODE') == "True":
        server_name = os.getenv("SERVER_NAME")
        if server_name:
            redirect_uri = url_for('auth.callback', _external=True, _scheme='https')
            print("Custom server name redirect: ", redirect_uri)
    return oauth.google.authorize_redirect(redirect_uri, nonce=nonce)

# OAuth 回調路由
@auth_bp.route('/callback')
def callback():
    try:
        token = oauth.google.authorize_access_token()
        if not token:
            return 'Failed to get token', 400
            
        # Get nonce from session
        nonce = session.get('nonce')
        if not nonce:
            return 'Invalid session', 400
            
        # Parse ID token with nonce
        user_info = oauth.google.parse_id_token(token, nonce=nonce)
        if not user_info:
            return 'Failed to get user info', 400

        user = User(
            id=user_info['sub'],
            email=user_info['email'],
            name=user_info.get('name', user_info['email'])
        )
        
        users[user.id] = user
        login_user(user)
        
        # 為使用者建立個人資料夾
        user_folder = os.path.join('users', secure_filename(user.id))
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            
        # 儲存使用者的 Google 電子郵件
        user_settings = load_user_settings(user.id)
        user_settings['email'] = user_info['email']
        save_user_settings(user.id, user_settings)
        
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Auth callback error: {str(e)}")
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
    ai_enabled = user_settings.get('ai_enabled', default("ai_enabled"))
    email_report_enabled = user_settings.get('email_report_enabled', default('email_report_enabled'))
    response = make_response(render_template('profile.html', ai_enabled=ai_enabled, email_report_enabled=email_report_enabled))
    # Add cache control headers to prevent caching
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
                return f'Binding request sent to {email}', 200
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

        # Get user settings for the current user
        user_settings = load_user_settings(current_user.id)
        bound_accounts = user_settings.get('bound_accounts', [])
        if request_user_id not in bound_accounts:
            bound_accounts.append(request_user_id)
            user_settings['bound_accounts'] = bound_accounts
            save_user_settings(current_user.id, user_settings)

        # Get user settings for the request user
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

# 檢查使用者是否有權限存取檔案
def check_file_access(file_path):
    if not current_user.is_authenticated:
        return False
    
    user_folder = os.path.join('users', secure_filename(current_user.id))
    return file_path.startswith(user_folder)

# 取得使用者上傳資料夾路徑
def get_user_upload_folder(user_id):
    user_folder = os.path.join('users', secure_filename(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def load_user_settings(user_id):
    user_folder = os.path.join('users', secure_filename(user_id))
    settings_file = os.path.join(user_folder, 'settings.json')
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            return settings
    except FileNotFoundError:
        return {}

def save_user_settings(user_id, settings):
    user_folder = os.path.join('users', secure_filename(user_id))
    settings_file = os.path.join(user_folder, 'settings.json')
    with open(settings_file, 'w') as f:
        json.dump(settings, f)

binding_requests = []

def send_binding_request(email, current_email):
    # TODO: Implement sending binding request logic (e.g., using email)
    print(f"Binding request sent to {email} from user {current_email}")
    return True

def store_binding_request(email, user_id, current_email):
    binding_requests.append({'email': email, 'request_user_id': user_id, 'request_user_email': current_email})
    print(f"Binding request stored for {email} from user {current_email}")
    return True

def get_user_by_id(user_id):
    users_dir = os.path.join(os.getcwd(), 'users')
    user_folder = os.path.join(users_dir, user_id)
    if os.path.isdir(user_folder):
        settings_file = os.path.join(user_folder, 'settings.json')
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    # Create a User object
                    user = User(
                        id=user_id,
                        email=settings.get('email'),
                        name=settings.get('name', settings.get('email'))
                    )
                    return user
            except (FileNotFoundError, json.JSONDecodeError):
                return None
    return None

def get_user_by_email(email):
    users_dir = os.path.join(os.getcwd(), 'users')
    for user_id in os.listdir(users_dir):
        user_folder = os.path.join(users_dir, user_id)
        if os.path.isdir(user_folder):
            settings_file = os.path.join(user_folder, 'settings.json')
            if os.path.exists(settings_file):
                try:
                    with open(settings_file, 'r') as f:
                        settings = json.load(f)
                        if settings.get('email') == email:
                            # Create a User object
                            user = User(
                                id=user_id,
                                email=settings.get('email'),
                                name=settings.get('name', settings.get('email'))
                            )
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

            # Remove current user from the other account's bound accounts
            other_user_settings = load_user_settings(account_id_to_remove)
            other_bound_accounts = other_user_settings.get('bound_accounts', [])
            if current_user.id in other_bound_accounts:
                other_bound_accounts.remove(current_user.id)
                other_user_settings['bound_accounts'] = other_bound_accounts
                save_user_settings(account_id_to_remove, other_user_settings)

    return redirect(url_for('auth.binding'))
