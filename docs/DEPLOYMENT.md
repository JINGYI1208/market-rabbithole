# Streamlit Community Cloud Deployment

## Render free-tier memory mode

Use `pip install -r requirements.txt` as the build dependency step. The deployment requirements intentionally exclude `sentence-transformers`, Transformers, and PyTorch. With Render's standard `RENDER=true` environment flag, `RABBITHOLE_RETRIEVAL_BACKEND=auto` resolves to sparse TF-IDF lexical retrieval, so a Live Agent request does not load the embedding model.

The backend can also be pinned explicitly with `RABBITHOLE_RETRIEVAL_BACKEND=tfidf`. For local MiniLM retrieval, install the explicit optional dependency with `pip install -e '.[semantic]'`; local `auto` mode then continues to select Sentence Transformers.

This removes the largest known memory consumer but does not guarantee that every workload remains below 512 MB; Streamlit, NumPy, PDF parsing, concurrent requests, and unusually large uploads still consume memory. Keep concurrency and upload limits conservative.

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
RABBITHOLE_RETRIEVAL_BACKEND = "auto"
```

The password is a lightweight demo gate, not production authentication. Keep concurrency at one and conservative search budgets for a public portfolio deployment. Rotate provider keys immediately if they are ever exposed.

## Operational notes

- The first live local-RAG request may download the configured sentence-transformer model and can be slow.
- Sentence Transformers and PyTorch are absent from deployment requirements when using the documented Render build. Public Demo never instantiates either retrieval backend.
- Uploads are request-scoped and not intentionally persisted, but a production system would still need formal retention, encryption, access-control, and provider data-processing policies.
- Community Cloud replicas are ephemeral; do not rely on local files for durable state.
- The repository validates local startup and HTTP behavior. Cloud behavior should be smoke-tested after deployment before sharing the URL.
