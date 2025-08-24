# Mahonia

Contributions and bug reports are welcome!

> [!IMPORTANT]
> ### First time setup
>
> - Install [uv](https://github.com/astral-sh/uv)
> - Initialize the environment:
>   ```
>   uv sync --locked --all-extras --dev
>   ```

### Formatting
```
uv run hatch run format
```
> [!NOTE]
> VSCode is setup to do this for you on save - feel free to add more editors.

### Linting
```
uv run hatch run lint
```
> [!NOTE]
> VSCode is setup to to run these LSPs in the background - feel free to add more
> editors.

### Tests
```
uv run hatch run test
```

### All
```
uv run hatch run all
```
