from rest_framework.views import APIView
from rest_framework import permissions, status
from rest_framework.response import Response


class UserInfo(APIView):

    permission_classes = (permissions.IsAuthenticated,)

    def get(self, request):
        """This endpoint provides infomation to the Oauth2 external services."""
        name = request.user.name
        last_name = request.user.lastname
        full_name = name + " " + last_name

        if request.user.is_staff and request.user.is_superuser:
            grafana_role = "Admin"
        elif request.user.is_staff:
            grafana_role = "Editor"
        else:
            grafana_role = "Viewer"

        payload = {
            "name": full_name,
            "email": request.user.email,
            "grafana_role": grafana_role,
            "is_staff": request.user.is_staff,
            "is_superuser": request.user.is_superuser,
        }

        return Response(payload, status=status.HTTP_200_OK)
