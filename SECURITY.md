# Security

Verilume stores user data locally by default under `~/.verilume`.

## Secrets

Do not commit Hugging Face tokens, web search provider API keys, `.env`, `.streamlit/secrets.toml`, `~/.verilume/config.env`, uploaded documents, or Chroma database files. The app masks tokens in the UI and `verilume config` output.

## Reporting

Report security issues privately.

Do not create public GitHub issues for vulnerabilities, leaked secrets, or exploit details.

Contact:

```text
security@ecosveri.dev
```

Until the EcosVeri website and private intake process are live, use GitHub private security advisories if available. If private advisories are unavailable, contact the maintainer through GitHub before publishing details.
