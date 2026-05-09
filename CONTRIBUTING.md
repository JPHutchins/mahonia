# Mahonia

Contributions and bug reports are welcome!

> [!IMPORTANT]
> ### First time setup
>
> - Install [uv](https://github.com/astral-sh/uv)
> - Initialize the environment:
>   ```
>   uv sync --locked --all-extras --all-packages --dev
>   ```

### Tasks

Tasks are defined in [`tasks.py`](tasks.py) and run with [camas](https://github.com/JPHutchins/camas):
```
uv run camas --help
```

> [!NOTE]
> VSCode is setup to format on save and run LSPs in the background — feel free
> to add more editors.
