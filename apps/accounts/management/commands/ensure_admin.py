from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = '确保管理员账户存在（默认 admin/admin123）'

    def add_arguments(self, parser):
        parser.add_argument('--username', default='admin')
        parser.add_argument('--password', default='admin123')
        parser.add_argument('--email', default='admin@dpds.local')

    def handle(self, *args, **options):
        username = options['username']
        password = options['password']
        email = options['email']

        user, created = User.objects.get_or_create(
            username=username,
            defaults={'email': email, 'is_staff': True, 'is_superuser': True},
        )
        if not created:
            user.is_staff = True
            user.is_superuser = True

        user.set_password(password)
        user.save()

        action = '已创建' if created else '已重置密码'
        self.stdout.write(self.style.SUCCESS(
            f'{action}管理员账户: {username} / {password}'
        ))
