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

# 使用者緩存 (一個簡單的記憶體緩存，用於減少檔案讀取)
users = {}

def init_auth(app):
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
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
    # Flask-Login 在每個請求開始時調用此函數，以獲取當前用戶物件
    if user_id in users:
        return users[user_id]
    # 如果不在緩存中，嘗試從檔案加載
    user = get_user_by_id(user_id)
    if user:
        users[user_id] = user
    return user

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
            redirect_uri = url_for('auth.callback', _external=True, _scheme='https')
    return oauth.google.authorize_redirect(redirect_uri, nonce=nonce)

# OAuth 回調路由 - 這是新用戶註冊的關鍵點
@auth_bp.route('/callback')
def callback():
    try:
        token = oauth.google.authorize_access_token()
        if not token:
            return 'Failed to get token', 400
            
        nonce = session.get('nonce')
        if not nonce:
            return 'Invalid session', 400
            
        user_info = oauth.google.parse_id_token(token, nonce=nonce)
        if not user_info:
            return 'Failed to get user info', 400

        user = User(
            id=user_info['sub'],
            email=user_info['email'],
            name=user_info.get('name', user_info['email'])
        )
        
        # 將用戶物件存入緩存並登入
        users[user.id] = user
        login_user(user)
        
        # *** 關鍵步驟：為新用戶或回訪用戶確保資料夾和設定檔存在 ***
        user_folder = get_user_upload_folder(user.id) # 這會自動創建資料夾
        user_settings = load_user_settings(user.id)
        
        # 儲存或更新用戶的基本資訊
        user_settings['email'] = user_info['email']
        user_settings['name'] = user.name
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
    return render_template('profile.html')

# 帳戶角色設定
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

# --- 帳戶綁定相關路由 ---

@auth_bp.route('/binding/add', methods=['GET', 'POST'])
@login_required
def account_binding():
    user_settings = load_user_settings(current_user.id)
    account_role = user_settings.get('account_role')

    if account_role != 'general':
        return '您沒有權限存取此頁面', 403

    if request.method == 'POST':
        email = request.form.get('email').lower().strip()
        
        # 檢查是否輸入自己的 email
        if email == current_user.email:
            return '您不能綁定自己的帳戶。', 400

        # *** 關鍵步驟：查找用戶是否存在 ***
        user_to_bind = get_user_by_email(email)
        
        if user_to_bind:
            # 檢查是否已經綁定或已發送請求
            sent_requests = [req for req in binding_requests if req['email'] == email and req['request_user_id'] == current_user.id]
            if sent_requests:
                return '您已經向此用戶發送過綁定請求。', 400
            
            user_settings = load_user_settings(current_user.id)
            if user_to_bind.id in user_settings.get('bound_accounts', []):
                return '您已經與此用戶綁定。', 400

            # 發送和儲存請求
            if send_binding_request(email, current_user.email):
                store_binding_request(email, current_user.id, current_user.email)
                return redirect(url_for('auth.binding'))
            else:
                return '發送綁定請求失敗。', 500
        else:
            # *** 這裡就是您看到錯誤的地方 ***
            return '系统中不存在此電子郵件對應的用戶。請確認對方已使用此 Google 帳戶登入過本系統。', 400

    return render_template('add_binding.html')

@auth_bp.route('/binding')
@login_required
def binding():
    user_id = current_user.id
    user_email = current_user.email
    
    # 過濾出與當前用戶相關的請求
    received_requests = [req for req in binding_requests if req['email'] == user_email]
    sent_requests = [req for req in binding_requests if req['request_user_id'] == user_id]
    
    user_settings = load_user_settings(user_id)
    bound_account_ids = user_settings.get('bound_accounts', [])
    
    # 獲取已綁定帳戶的 email
    bound_account_emails = []
    for account_id in bound_account_ids:
        user = get_user_by_id(account_id)
        if user:
            bound_account_emails.append(user.email)
            
    return render_template('binding.html', 
                           received_requests=received_requests, 
                           sent_requests=sent_requests, 
                           bound_accounts=bound_account_emails)

@auth_bp.route('/binding/accept/<request_user_id>')
@login_required
def accept_binding(request_user_id):
    global binding_requests
    # 找到對應的請求
    request_to_process = next((req for req in binding_requests if req['email'] == current_user.email and req['request_user_id'] == request_user_id), None)
    
    if request_to_process:
        # 1. 在當前用戶的設定中，加入請求者的 ID
        user_settings = load_user_settings(current_user.id)
        bound_accounts = user_settings.get('bound_accounts', [])
        if request_user_id not in bound_accounts:
            bound_accounts.append(request_user_id)
            user_settings['bound_accounts'] = bound_accounts
            save_user_settings(current_user.id, user_settings)

        # 2. 在請求者的設定中，加入當前用戶的 ID
        request_user_settings = load_user_settings(request_user_id)
        request_user_bound_accounts = request_user_settings.get('bound_accounts', [])
        if current_user.id not in request_user_bound_accounts:
            request_user_bound_accounts.append(current_user.id)
            request_user_settings['bound_accounts'] = request_user_bound_accounts
            save_user_settings(request_user_id, request_user_settings)
            
        # 3. 從請求列表中移除已處理的請求
        binding_requests.remove(request_to_process)

    return redirect(url_for('auth.binding'))

@auth_bp.route('/binding/reject/<request_user_id>')
@login_required
def reject_binding(request_user_id):
    global binding_requests
    binding_requests = [req for req in binding_requests if not (req['email'] == current_user.email and req['request_user_id'] == request_user_id)]
    return redirect(url_for('auth.binding'))

@auth_bp.route('/binding/withdraw/<email>')
@login_required
def withdraw_binding(email):
    global binding_requests
    binding_requests = [req for req in binding_requests if not (req['email'] == email and req['request_user_id'] == current_user.id)]
    return redirect(url_for('auth.binding'))

@auth_bp.route('/binding/remove/<email>')
@login_required
def remove_binding(email):
    user_to_remove = get_user_by_email(email)
    if not user_to_remove:
        return redirect(url_for('auth.binding'))

    account_id_to_remove = user_to_remove.id

    # 1. 從當前用戶的綁定列表中移除對方
    user_settings = load_user_settings(current_user.id)
    bound_accounts = user_settings.get('bound_accounts', [])
    if account_id_to_remove in bound_accounts:
        bound_accounts.remove(account_id_to_remove)
        user_settings['bound_accounts'] = bound_accounts
        save_user_settings(current_user.id, user_settings)

    # 2. 從對方的綁定列表中移除當前用戶
    other_user_settings = load_user_settings(account_id_to_remove)
    other_bound_accounts = other_user_settings.get('bound_accounts', [])
    if current_user.id in other_bound_accounts:
        other_bound_accounts.remove(current_user.id)
        other_user_settings['bound_accounts'] = other_bound_accounts
        save_user_settings(account_id_to_remove, other_user_settings)

    return redirect(url_for('auth.binding'))

# --- 輔助函數 ---

def get_user_upload_folder(user_id):
    static_folder = 'static'
    user_folder = os.path.join(static_folder, 'users', secure_filename(str(user_id)))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def load_user_settings(user_id):
    user_folder = get_user_upload_folder(user_id)
    settings_file = os.path.join(user_folder, 'settings.json')
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_user_settings(user_id, settings):
    user_folder = get_user_upload_folder(user_id)
    settings_file = os.path.join(user_folder, 'settings.json')
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)

# 使用一個簡單的列表來模擬資料庫儲存綁定請求
binding_requests = []

def send_binding_request(email, current_email):
    print(f"模擬發送郵件通知給 {email}，告知來自 {current_email} 的綁定請求。")
    return True

def store_binding_request(email, user_id, current_email):
    # 避免重複發送
    if not any(req['email'] == email and req['request_user_id'] == user_id for req in binding_requests):
        binding_requests.append({'email': email, 'request_user_id': user_id, 'request_user_email': current_email})
    print(f"Binding request stored for {email} from user {current_email}")
    return True

def get_user_by_id(user_id):
    user_folder = get_user_upload_folder(user_id)
    settings_file = os.path.join(user_folder, 'settings.json')
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                user = User(
                    id=str(user_id),
                    email=settings.get('email'),
                    name=settings.get('name', settings.get('email'))
                )
                return user
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    return None

def get_user_by_email(email):
    users_base_dir = os.path.join('static', 'users')
    if not os.path.exists(users_base_dir):
        return None
        
    for user_id in os.listdir(users_base_dir):
        user_folder = os.path.join(users_base_dir, user_id)
        if os.path.isdir(user_folder):
            settings_file = os.path.join(user_folder, 'settings.json')
            if os.path.exists(settings_file):
                try:
                    with open(settings_file, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                        if settings.get('email') == email:
                            user = User(
                                id=user_id,
                                email=settings.get('email'),
                                name=settings.get('name', settings.get('email'))
                            )
                            return user
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
    return None