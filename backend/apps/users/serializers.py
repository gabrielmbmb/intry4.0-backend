from rest_framework import serializers
from backend.apps.users.models import User


class RegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(max_length=128, min_length=8, write_only=True)
    password2 = serializers.CharField(max_length=128, min_length=8, write_only=True)

    class Meta:
        model = User
        fields = ["email", "username", "name", "lastname", "password", "password2"]

    def validate(self, data):
        password = data.get("password", None)
        password2 = data.get("password2", None)

        if not password or not password2:
            raise serializers.ValidationError("Please, enter a password and confirm it")

        if password != password2:
            raise serializers.ValidationError("The passwords did not match!")

        return data

    def create(self, validated_data):
        # There is no need to save the confirmation password
        validated_data.pop("password2", None)
        return User.objects.create_user(**validated_data)


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(max_length=128, min_length=8, write_only=True)

    class Meta:
        model = User
        fields = ["email", "username", "name", "lastname", "password"]

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
