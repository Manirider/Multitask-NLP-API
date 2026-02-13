def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

def test_predict_sentiment_validation_error(client):
    response = client.post("/predict/sentiment", json={})
    assert response.status_code == 422

def test_predict_ner_validation_error(client):
    response = client.post("/predict/ner", json={})
    assert response.status_code == 422

def test_predict_qa_validation_error(client):
    response = client.post("/predict/qa", json={})
    assert response.status_code == 422

def test_predict_sentiment_model_not_loaded(client):
    response = client.post("/predict/sentiment",
                           json={"text": "This is great"})
    assert response.status_code == 503

def test_predict_qa_model_not_loaded(client):
    response = client.post(
        "/predict/qa", json={"context": "The cat sat.", "question": "What sat?"})
    assert response.status_code == 503
