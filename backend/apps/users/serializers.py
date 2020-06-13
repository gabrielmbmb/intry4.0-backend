from rest_framework import serializers
from backend.apps.users.models import User


def check_password(password, password2):
    """
    Checks if the password and password confirmation are the same and are provided.

    Raises:
        rest_framework.serializes.ValidationError: if the passwords are not the same or
            any is not provided.
    """
    if not password or not password2 or password != password2:
        raise serializers.ValidationError("The passwords did not match!")


class RegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(max_length=128, min_length=8, write_only=True)
    password2 = serializers.CharField(max_length=128, min_length=8, write_only=True)

    class Meta:
        model = User
        fields = ["email", "username", "name", "lastname", "password", "password2"]

    def validate(self, data):
        password = data.get("password", None)
        password2 = data.get("password2", None)
        check_password(password, password2)

        return data

    def create(self, validated_data):
        # There is no need to save the confirmation password
        validated_data.pop("password2", None)
        return User.objects.create_user(**validated_data)


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(max_length=128, min_length=8, write_only=True)

    class Meta:
        model = User
        ref_name = None
        fields = [
            "id",
            "email",
            "username",
            "name",
            "lastname",
            "password",
            "is_active",
            "is_staff",
        ]

    def update(self, instance, validated_data):
        """Updates user data."""
        password = validated_data.pop("password", None)

        # iterate over the validated data (fields)
        for (key, value) in validated_data.items():
            setattr(instance, key, value)

        # password should not be set with `setattr`. Django provides the function
        # `set_password` which handles all security stuff for us.
        if password is not None:
            instance.set_password(password)

        instance.save()

        return instance


class ChangePasswordSerializer(serializers.Serializer):
    """Serializer for user password change."""

    model = User
    old_password = serializers.CharField(max_length=128, min_length=8)
    password = serializers.CharField(max_length=128, min_length=8)
    password2 = serializers.CharField(max_length=128, min_length=8)

    class Meta:
        ref_name = None

    def validate(self, data):
        password = data.get("password", None)
        password2 = data.get("password2", None)
        check_password(password, password2)

        return data

    def update(self, instance, validated_data):
        """Changes the user password."""
        password = validated_data.get("password", None)
        instance.set_password(password)
        instance.save()

        return instance
