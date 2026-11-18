"""Model and dataset cards, and pasted links."""


def test_catalog_lists_cards(client):
    cat = client.get("/api/catalog").json()
    assert {m["id"] for m in cat["models"]} >= {"persistence", "linear"}
    assert all(d["status"]["state"] == "ready" for d in cat["datasets"])
    assert cat["problems"] == []


def test_add_and_remove_cards(client):
    assert client.post("/api/models", json={"id": "Bad ID", "name": "x", "adapter": "persistence"}).status_code == 422
    card = {"id": "extra", "name": "Extra", "source": {"type": "url", "url": "https://example.org/x.npy"}}
    added = client.post("/api/datasets", json=card)
    assert added.status_code == 201 and added.json()["status"]["state"] == "download"
    assert client.post("/api/datasets", json=card).status_code == 409  # same id twice
    assert client.delete("/api/datasets/extra").status_code == 200
    assert client.delete("/api/datasets/extra").status_code == 404


def test_parse_links(client):
    def parse(link):
        return client.post("/api/parse-link", json={"link": link}).json()

    assert parse("https://huggingface.co/mahindrautela/MORPH/blob/main/models/FM/morph-Ti-FM-max_ar1_ep225.pth") == {
        "type": "huggingface", "repo_id": "mahindrautela/MORPH", "repo_type": "model",
        "subfolder": "models/FM", "filename": "morph-Ti-FM-max_ar1_ep225.pth"}
    hf_dataset = parse("https://huggingface.co/datasets/camlab-ethz/FNS-KF/blob/main/solution_0.nc")
    assert hf_dataset["repo_type"] == "dataset" and hf_dataset["filename"] == "solution_0.nc"
    assert parse("https://github.com/lanl/MORPH/blob/package-morph/docs/user-guide.md") == {
        "type": "github", "repo_id": "lanl/MORPH", "repo_type": "model",
        "revision": "package-morph", "filename": "docs/user-guide.md"}
    assert parse("https://example.org/data.h5")["type"] == "url"
    assert parse("/data/x.npy")["type"] == "local"
