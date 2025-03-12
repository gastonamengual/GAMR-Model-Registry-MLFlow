import requests

from app.settings import Settings


def get_service_id() -> str:
    url = (
        "https://api.render.com/v1/services?includePreviews=true&limit=20"  # Render API
    )

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {Settings.RENDER_API_TOKEN}",
    }

    response = requests.get(url, headers=headers)
    response_id: str = response.json()[0]["service"]["id"]
    return response_id


def deploy_service(service_id: str) -> None:
    url = f"https://api.render.com/v1/services/{service_id}/deploys"  # Render API

    payload = {"clearCache": "do_not_clear"}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {Settings.RENDER_API_TOKEN}",
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.ok:
        pass


if __name__ == "__main__":
    service_id = get_service_id()
    deploy_service(service_id)
