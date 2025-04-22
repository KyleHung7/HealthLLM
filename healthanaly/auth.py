import os
from flask import Flask, redirect, url_for, session, request, Blueprint, render_template, make_response
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from authlib.integrations.flask_client import OAuth
import json
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
        user_folder = os.path.join('uploads', secure_filename(user.id))
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        
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
    response = make_response(render_template('profile.html'))
    # Add cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# 檢查使用者是否有權限存取檔案
def check_file_access(file_path):
    if not current_user.is_authenticated:
        return False
    
    user_folder = os.path.join('uploads', secure_filename(current_user.id))
    return file_path.startswith(user_folder)

# 取得使用者上傳資料夾路徑
def get_user_upload_folder():
    if not current_user.is_authenticated:
        return None
    
    user_folder = os.path.join('uploads', secure_filename(current_user.id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    return user_folder
