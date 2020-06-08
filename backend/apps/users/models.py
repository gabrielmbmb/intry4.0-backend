from django.db import models
from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin,
)


class UserManager(BaseUserManager):
    """Custom user manager required to create custom users."""

    def create_user(self, username, email, password=None, **kwargs):
        """
        Create and return a `User`.
        """
        if username is None:
            raise TypeError("Users must have a username.")

        if email is None:
            raise TypeError("Users must have an email address.")

        user = self.model(
            username=username, email=self.normalize_email(email), **kwargs
        )
        user.set_password(password)
        user.save()

        return user

    def create_superuser(self, username, email, password, **kwargs):
        """
        Create and return a `User` with superuser role.
        """
        if password is None:
            raise TypeError("Superusers must have a passwords.")

        user = self.create_user(
            username=username,
            email=self.normalize_email(email),
            password=password,
            **kwargs
        )
        user.is_superuser = True
        user.is_staff = True
        user.save()

        return user


class User(AbstractBaseUser, PermissionsMixin):
    """
    Custom user class which overrides the default of Django.

    Attributes:
        username (django.db.models.CharField): human-readable unique identifier that can
            be used to represent the `User` in the UI.
        email (django.db.models.EmailField): user email used for logging and for
            contacting the user.
        name (django.db.models.CharField): user real name.
        lastname (django.db.models.CharField): user real lastname.
        is_active(django.db.models.BooleanField): indicates if the account is active. It
            can be used when a user wants to delete its account, but its account data
            has to remain in storage. Defaults to True.
        is_staff(django.db.models.BooleanField): flag expected by Django to know who can
            log into the admin site. Defaults to False.
    """

    username = models.CharField(db_index=True, max_length=255, unique=True)
    email = models.EmailField(db_index=True, unique=True)
    name = models.CharField(db_index=True, max_length=255)
    lastname = models.CharField(db_index=True, max_length=255)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    # Indicates Django which field is going to be used to perform the logging.
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username", "name", "lastname"]

    # Indicates Django that `UserManager` have to be used to manage `User` objects.
    objects = UserManager()

    def __str__(self):
        """
        String representation of this class instance.

        Returns:
            str: string class representation.
        """
        return self.email

    def get_full_name(self):
        """
        Required method for Django.

        Returns:
            str: user full name
        """
        return self.name + " " + self.lastname

    def get_short_name(self):
        """
        Required method for Django.

        Returns:
            str: user short name
        """
        return self.name
