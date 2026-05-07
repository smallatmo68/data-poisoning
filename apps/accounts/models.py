from django.contrib.auth.models import User
from django.db import models


class UserProfile(models.Model):
    """扩展 Django 内置 User 的用户信息。"""

    ROLE_CHOICES = [
        ('admin', '管理员'),
        ('analyst', '分析师'),
        ('viewer', '只读用户'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField('角色', max_length=20, choices=ROLE_CHOICES, default='analyst')
    department = models.CharField('部门', max_length=100, blank=True)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        db_table = 'dpds_user_profile'
        verbose_name = '用户资料'
        verbose_name_plural = '用户资料'

    def __str__(self):
        return f'{self.user.username} ({self.role})'
