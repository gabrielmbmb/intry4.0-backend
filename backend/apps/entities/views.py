from typing import List
from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from backend.apps.entities.clients import OrionClient


def get_entity_attributes(entity: dict) -> List[str]:
    """Get a list with the attributes of the entity.

    Args:
        entity(:obj:`dict`): entity.

    Returns:
        list: containing the attributes.
    """
    keys = entity.keys()
    not_attributes = ["id", "type"]
    attributes = [attr for attr in keys if attr not in not_attributes]
    return attributes


class EntitiesView(APIView):
    """View to get a list of entities from FIWARE."""

    permission_classes = (permissions.IsAuthenticated,)

    def get(self, request, **kwargs):
        urn = kwargs.get("urn")
        client = OrionClient()

        if urn is not None:
            entity = client.get_entities(urn)
            data = {
                "id": entity["id"],
                "type": entity["type"],
                "attributes:": get_entity_attributes(entity),
            }

        else:
            entities = client.get_entities()
            data = []
            for entity in entities:
                data.append(
                    {
                        "id": entity["id"],
                        "type": entity["type"],
                        "attributes:": get_entity_attributes(entity),
                    }
                )

        return Response(data, status=status.HTTP_200_OK)
