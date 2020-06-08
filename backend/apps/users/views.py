from rest_framework import permissions, status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response

# from rest_framework.decorators import action
from backend.apps.users.serializers import (
    RegistrationSerializer,
    UserSerializer,
    ChangePasswordSerializer,
)
from backend.apps.users.models import User


class UserViewSet(viewsets.ModelViewSet):
    """This viewset automatically provides all actions for users."""

    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (permissions.IsAdminUser,)

    def create(self, request, *args, **kwargs):
        # validate incoming data from user registration
        serializer = RegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # data is valid so create a new user
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class UpdateUserPasswordView(APIView):
    """View to update the user password."""

    permission_classes = (permissions.IsAuthenticated,)

    def get_object(self, queryset=None):
        return self.request.user

    def post(self, request, format=None):
        self.object = self.get_object()
        serializer = ChangePasswordSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        if not self.object.check_password(serializer.data.get("old_password", None)):
            return Response(
                {"old_password": ["Wrong password."]},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # old password was correct and passwords provided did match
        self.object.set_password(serializer.data.get("password", None))
        self.object.save()

        return Response(
            {"detail": "The password has been updated!"}, status=status.HTTP_200_OK
        )
