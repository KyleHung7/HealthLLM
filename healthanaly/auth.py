import os
from flask import redirect, url_for, session, Blueprint, render_template, make_response, request
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from authlib.integrations.flask_client import OAuth
import json
from constants.default_settings import default
from werkzeug.utils import secure_filename

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

# 檢查使用者是否有權限存取檔案
def check_file_access(file_path):
    if not current_user.is_authenticated:
        return False
    
    user_folder = os.path.join('users', secure_filename(current_user.id))
    return file_path.startswith(user_folder)

# 取得使用者上傳資料夾路徑
def get_user_upload_folder():
    if not current_user.is_authenticated:
        return None
    
    user_folder = os.path.join('users', secure_filename(current_user.id))
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
