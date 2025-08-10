# test/test_flask_app.py

import pytest
from flask3.app import create_app

@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"<html" in response.data or b"<!DOCTYPE html>" in response.data

def test_video_feed_route(client):
    response = client.get('/video_feed')
    assert response.status_code == 200
    assert response.mimetype == 'multipart/x-mixed-replace'
    assert b'TestImageData' in response.data
