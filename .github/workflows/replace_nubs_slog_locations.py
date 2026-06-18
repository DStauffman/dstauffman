"""Replace the locations of slog and nubs."""

with open("pyproject.toml", "rt", encoding="utf-8") as file:
    text = file.read().replace('"../nubs"', '"../nubs_repo"').replace('"../slog"', '"../slog_reoo"')
with open("pyproject.toml", "wt", encoding="utf-8") as file:
    file.write(text)
    