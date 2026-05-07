from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import LoginView, LogoutView, ProfileView, UserListView
from .admin_views import AdminUserListView, AdminUserToggleView, AdminDashboardView, AdminAuditLogView

urlpatterns = [
    path('login/', LoginView.as_view(), name='auth-login'),
    path('profile/', ProfileView.as_view(), name='auth-profile'),
    path('logout/', LogoutView.as_view(), name='auth-logout'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token-refresh'),
    path('users/', UserListView.as_view(), name='auth-users'),
    path('admin/users/', AdminUserListView.as_view(), name='admin-users'),
    path('admin/users/<int:user_id>/toggle/', AdminUserToggleView.as_view(), name='admin-user-toggle'),
    path('admin/dashboard/', AdminDashboardView.as_view(), name='admin-dashboard'),
    path('admin/audit-logs/', AdminAuditLogView.as_view(), name='admin-audit-logs'),
]
