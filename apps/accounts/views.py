from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .serializers import LoginSerializer, UserSerializer


def ok(data=None, msg='OK'):
    return Response({'code': 0, 'msg': msg, 'data': data})


def err(msg, code=4001, status_code=status.HTTP_400_BAD_REQUEST):
    return Response({'code': code, 'msg': msg, 'data': None}, status=status_code)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            msgs = ' '.join(str(v) for vs in serializer.errors.values() for v in vs)
            return err(msgs)

        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)
        return ok({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': UserSerializer(user).data,
        })


class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return ok(UserSerializer(request.user).data)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get('refresh')
            token = RefreshToken(refresh_token)
            token.blacklist()
        except Exception:
            pass
        return ok(msg='已退出登录')


class UserListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        users = User.objects.select_related('profile').filter(is_active=True).order_by('-date_joined')
        data = []
        for u in users:
            profile = getattr(u, 'profile', None)
            data.append({
                'id': u.id,
                'username': u.username,
                'email': u.email,
                'is_staff': u.is_staff,
                'role': profile.role if profile else 'analyst',
                'department': profile.department if profile else '',
            })
        return ok(data)
