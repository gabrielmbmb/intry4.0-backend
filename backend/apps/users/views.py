from rest_framework import permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from backend.apps.users.serializers import RegistrationSerializer, UserSerializer
from backend.apps.users.models import User


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    This viewset automatically provides `list` and `detail` actions of users.
    """

    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    @action(detail=False, methods=["POST"], permission_classes=[permissions.AllowAny])
    def register(self, request):
        # validate incoming data from user registration
        serializer = RegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # data is valid so create a new user
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)