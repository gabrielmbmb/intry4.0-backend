from rest_framework_extensions.routers import ExtendedDefaultRouter
from backend.apps.datamodel import views

router = ExtendedDefaultRouter()
router.register(r"predictions", views.DatamodelPredictionViewSet)
router.register(r"datamodels", views.DataModelViewSet).register(
    r"predictions",
    views.DatamodelPredictionViewSet,
    basename="datamodel-predictions",
    parents_query_lookups=["datamodel"],
)

urlpatterns = router.urls
