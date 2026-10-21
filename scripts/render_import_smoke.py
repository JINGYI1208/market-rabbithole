"""Render dependency smoke check; imports and constructs clients without API calls."""


def check_zhipu_client_import() -> str:
    from zhipuai import ZhipuAI

    client = ZhipuAI(api_key="import-smoke-test-not-a-real-key")
    return type(client).__name__


if __name__ == "__main__":
    print(f"Zhipu import smoke: {check_zhipu_client_import()}")
