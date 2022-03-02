from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    # When
    response = client.get("http://localhost:8001/api/v1/health")

    # Then
    print(response)
    assert response.status_code == 200
