import pymysql
pymysql.install_as_MySQLdb()

try:
    from .celery import app as celery_app
    __all__ = ('celery_app',)
except ImportError:
    pass
