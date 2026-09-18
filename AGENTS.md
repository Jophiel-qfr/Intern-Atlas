# Project rules

- Keep code in `app`; keep runtime data, databases, PDFs, caches, and logs in configurable external directories.
- Do not create, modify, or delete the sibling `env` environment from project work.
- Never commit API keys, `.env` files, PDFs, generated databases, caches, or logs.
- Do not download large models or add large infrastructure dependencies without a separate request.
- Do not use destructive Git commands, force push, delete branches, or modify `main`.
- Preserve existing CLI/API behavior and relationship values when adding modules.
- Prefer small, independent modules for new analysis and external integrations.
- Put user-facing UI copy in the language layer; keep paper titles, model names, datasets, abbreviations, and quoted evidence in their original form.

