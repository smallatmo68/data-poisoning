"""
WSGI config for DataPoisoningDetection project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DataPoisoningDetection.settings')

application = get_wsgi_application()
