"""
URL configuration for DataPoisoningDetection / DPDS project.

路由设计：
  /api/auth/        → accounts（JWT 认证）
  /api/datasets/    → dpds_datasets（新版数据集管理）
  /api/legacy/      → datasets（原有 API，向后兼容）
  /api/preprocess/  → preprocessing
  /api/detection/   → detection
  /api/defense/     → defense
  /api/reports/     → reports
  /api/admin/       → audit / algorithm config
"""
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path
from django.views.generic import RedirectView, TemplateView

urlpatterns = [
    # favicon.ico
    path('favicon.ico', RedirectView.as_view(url=settings.STATIC_URL + 'favicon.ico', permanent=True)),

    # Django admin
    path('admin/', admin.site.urls),

    # ── 认证 ─────────────────────────────────────────────────────
    path('api/auth/', include('apps.accounts.urls')),

    # ── 数据集（新版）────────────────────────────────────────────
    path('api/datasets/', include('apps.dpds_datasets.urls')),

    # ── 预处理 ───────────────────────────────────────────────────
    path('api/preprocess/', include('apps.preprocessing.urls')),

    # ── 检测任务（新版）────────────────────────────────────────
    path('api/detection/', include('apps.detection.urls')),

    # ── 无害化防御 ───────────────────────────────────────────────
    path('api/defense/', include('apps.defense.urls')),

    # ── 报告 ─────────────────────────────────────────────────────
    path('api/reports/', include('apps.reports.urls')),

    # ── 审计日志（管理员，旧路由兼容）────────────────────────────
    path('api/admin/audit-logs-old/', include('apps.audit.urls')),

    # ── 前端 SPA 入口 ─────────────────────────────────────────────
    path('', TemplateView.as_view(template_name='index.html'), name='index'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
