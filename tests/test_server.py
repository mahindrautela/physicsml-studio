"""The page and its files are served with the right content types.

Browsers refuse to run JavaScript modules served as text/plain, which some
Windows setups would otherwise produce, leaving a blank page.
"""


def test_page_and_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert "javascript" in client.get("/js/app.js").headers["content-type"]
    assert "text/css" in client.get("/css/app.css").headers["content-type"]


def test_cards_with_non_ascii_text_round_trip(client):
    card = {"id": "unicode", "name": "Navier–Stokes 128×128", "adapter": "persistence",
            "description": "next = 2 × current − previous"}
    assert client.post("/api/models", json=card).status_code == 201
    saved = next(m for m in client.get("/api/catalog").json()["models"] if m["id"] == "unicode")
    assert (saved["name"], saved["description"]) == (card["name"], card["description"])
    client.delete("/api/models/unicode")
