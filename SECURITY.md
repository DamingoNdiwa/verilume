# Security

Verilume stores user data locally by default under `~/.verilume`.

## Supported Versions

Security fixes target the latest release. Older releases do not receive patches.

## Secrets

Do not commit Hugging Face tokens, web search provider API keys, `.env`, `.streamlit/secrets.toml`, `~/.verilume/config.env`, uploaded documents, or Chroma database files. The app masks tokens in the UI and `verilume config` output.

## Reporting

Report security issues privately. Do not create public GitHub issues for vulnerabilities, leaked secrets, or exploit details.

Use GitHub private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. If private reporting is unavailable, contact the maintainer through GitHub before publishing details.

We aim to acknowledge reports within 7 days.

A dedicated security contact address will be published when the EcosVeri website is live.
