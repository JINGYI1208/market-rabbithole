# Streamlit Community Cloud Deployment

## Public demo

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the main file to `streamlit_app.py`.
4. Deploy without secrets. The default is `RABBITHOLE_LIVE_ENABLED=0`, so only the three fixture-based recorded runs are available and no external API is called.

`requirements.txt`, `runtime.txt`, and `.streamlit/config.toml` provide the hosted runtime configuration.

## Optional live mode

Add these in the Streamlit app's **Secrets** settings, never in Git:

```toml
ZHIPUAI_API_KEY = "..."
TAVILY_API_KEY = "..."
ZHIPU_MODEL = "glm-4-flash"
RABBITHOLE_LIVE_ENABLED = "1"
RABBITHOLE_LIVE_PASSWORD = "a-long-random-password"
```

The password is a lightweight demo gate, not production authentication. Keep concurrency at one and conservative search budgets for a public portfolio deployment. Rotate provider keys immediately if they are ever exposed.

## Operational notes

- The first live local-RAG request may download the configured sentence-transformer model and can be slow.
- `sentence-transformers` and its Torch dependency may approach free-tier memory/startup limits. Public Demo does not instantiate the model.
- Uploads are request-scoped and not intentionally persisted, but a production system would still need formal retention, encryption, access-control, and provider data-processing policies.
- Community Cloud replicas are ephemeral; do not rely on local files for durable state.
- The repository validates local startup and HTTP behavior. Cloud behavior should be smoke-tested after deployment before sharing the URL.
