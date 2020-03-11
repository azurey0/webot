from app import app
from app import app, db
from app.models import User

# 实现 flask shell 上下文环境
@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User}