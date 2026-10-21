from scripts.render_import_smoke import check_zhipu_client_import


def test_render_can_instantiate_zhipu_client_without_request():
    assert check_zhipu_client_import() == "ZhipuAI"
