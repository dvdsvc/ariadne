# Environment configuration / 环境配置

Offline examples require no API credentials. For model-backed runs, set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` for the provider selected by the launcher. Data paths and execution options are listed in `.env.example`.

离线示例无需密钥。运行模型实验时，按所选服务配置 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`；数据路径和执行选项见示例文件。

## Local setup

```bash
cp .env.example .env
```

Fill in the supported variables in your local `.env`. The file is ignored by Git; keep the committed template's credential fields empty. For entrypoints that read process variables, add `--env-file .env` to the documented `uv run` command. See the repository's running guide for provider and data options.

在本地 `.env` 中填写所需变量，提交的示例文件保持空白。读取进程环境变量的入口可通过 `uv run --env-file .env ...` 载入配置，具体运行方式见对应说明。

## Repository check

With [Gitleaks](https://github.com/gitleaks/gitleaks) installed and on PATH:

```bash
uv run python scripts/check_secrets.py --history
```

This checks publication files and reachable Git history using the repository's detector configuration. Reports show locations without credential values.
