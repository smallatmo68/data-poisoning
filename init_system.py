"""
DPDS 数据投毒检测系统 - 初始化脚本
运行方式: python init_system.py
"""
import os
import sys
import subprocess


def run_cmd(cmd, cwd=None):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"命令执行失败，返回码: {result.returncode}")
    return result.returncode


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    manage_py = os.path.join(project_dir, 'manage.py')

    print("=" * 60)
    print("  DPDS 数据投毒检测系统 - 初始化")
    print("=" * 60)

    print("\n[1/6] 执行数据库迁移...")
    run_cmd(f'python "{manage_py}" makemigrations')
    run_cmd(f'python "{manage_py}" migrate')

    print("\n[2/6] 创建管理员账号...")
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DataPoisoningDetection.settings')
    sys.path.insert(0, project_dir)
    try:
        import django
        django.setup()
        from django.contrib.auth.models import User
        from apps.accounts.models import UserProfile
        admin, created = User.objects.get_or_create(
            username='admin',
            defaults={
                'is_staff': True,
                'is_superuser': True,
                'is_active': True,
                'email': 'admin@dpds.local',
            }
        )
        if created:
            admin.set_password('admin123')
            admin.save()
            print("  已创建管理员: admin / admin123")
        else:
            print("  管理员账号已存在，跳过")
        UserProfile.objects.get_or_create(user=admin, defaults={'role': 'admin'})
    except Exception as e:
        print(f"  创建管理员失败: {e}")

    print("\n[3/6] 创建普通测试用户...")
    try:
        user, created = User.objects.get_or_create(
            username='user',
            defaults={
                'is_staff': False,
                'is_active': True,
                'email': 'user@dpds.local',
            }
        )
        if created:
            user.set_password('user123')
            user.save()
            from apps.accounts.models import UserProfile
            UserProfile.objects.get_or_create(user=user, defaults={'role': 'analyst'})
            print("  已创建测试用户: user / user123")
        else:
            print("  测试用户已存在，跳过")
    except Exception as e:
        print(f"  创建测试用户失败: {e}")

    print("\n[4/6] 注册检测算法...")
    run_cmd(f'python "{manage_py}" register_algorithms')

    print("\n[5/6] 收集静态文件...")
    run_cmd(f'python "{manage_py}" collectstatic --noinput')

    print("\n[6/6] 创建媒体目录...")
    for d in ['media/uploads', 'media/preprocessed', 'media/clean', 'media/reports']:
        os.makedirs(os.path.join(project_dir, d), exist_ok=True)

    print("\n" + "=" * 60)
    print("  初始化完成！")
    print("=" * 60)
    print("\n测试账号:")
    print("  管理员: admin / admin123")
    print("  普通用户: user / user123")
    print("\n启动开发服务器:")
    print(f'  python "{manage_py}" runserver 0.0.0.0:8000')
    print("\n访问地址: http://localhost:8000")


if __name__ == '__main__':
    main()
