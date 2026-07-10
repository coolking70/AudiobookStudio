import project_store


def test_project_persists_confirmed_alias_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "OUTPUT_DIR", tmp_path)
    project = project_store.build_project_from_state(
        "book-one",
        "Book One",
        [{"speaker": "甲", "text": "你好"}],
        role_profiles={"甲": {}},
        alias_map={"阿甲": "甲"},
    )
    assert project.cast[0].aliases == ["阿甲"]

    project.cast[0].aliases.append("甲先生")
    project_store.save_project(project)
    loaded = project_store.load_project("book-one")
    assert loaded.cast[0].aliases == ["阿甲", "甲先生"]
    assert project_store.canonical_speaker(loaded, "甲先生") == "甲"
