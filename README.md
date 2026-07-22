# AI Notes MkDocs

AI Notes 是使用 MkDocs Material 建置的 AI 筆記網站。

線上網站：https://chioujryu.github.io/ai-notes-mkdocs/

## 環境需求

- uv
- Git

本專案使用 uv 管理 Python 版本、虛擬環境與套件相依。預設 Python 版本寫在 `.python-version`，目前為 Python 3.12。

## 安裝 uv

Linux / macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安裝後重新開啟終端機，確認 uv 可用：

```bash
uv --version
```

## 初次設定

Linux / macOS:

```bash
git clone https://github.com/chioujryu/my-mkdocs-AI-docs.git ai-notes-mkdocs
cd ai-notes-mkdocs
uv sync --locked
```

Windows PowerShell:

```powershell
git clone https://github.com/chioujryu/my-mkdocs-AI-docs.git ai-notes-mkdocs
cd ai-notes-mkdocs
uv sync --locked
```

`uv sync --locked` 會依照 `uv.lock` 建立 `.venv`，並安裝 MkDocs 與 Material theme。若本機沒有指定的 Python 版本，uv 會自動下載可用版本。

## 本機預覽

Linux / macOS:

```bash
uv run mkdocs serve -a 127.0.0.1:8000
```

Windows PowerShell:

```powershell
uv run mkdocs serve -a 127.0.0.1:8000
```

開啟瀏覽器：

```text
http://127.0.0.1:8000
```

修改 `docs/` 或 `mkdocs.yml` 後，MkDocs 會自動重新載入。

## 建置網站

Linux / macOS:

```bash
uv run mkdocs build --strict
```

Windows PowerShell:

```powershell
uv run mkdocs build --strict
```

建置結果會輸出到 `site/`。`site/` 是產物目錄，不需要提交到 Git。

## 常用指令

```bash
# 同步鎖定版本的環境
uv sync --locked

# 啟動本機預覽
uv run mkdocs serve -a 127.0.0.1:8000

# 嚴格模式建置，CI 也使用這個指令
uv run mkdocs build --strict

# 更新依賴版本並重建 uv.lock
uv lock --upgrade

# 檢查目前 uv 管理的 Python
uv python list --only-installed
```

## 專案結構

```text
.
├── docs/                  # Markdown 內容與圖片資產
├── mkdocs.yml             # MkDocs 導覽、主題與 Markdown extension 設定
├── pyproject.toml         # uv project 與 Python dependencies
├── uv.lock                # 跨平台鎖定檔
├── requirements.txt       # Read the Docs / pip 相容 fallback
└── .github/workflows/     # CI 與 GitHub Pages 部署
```

## 新增或修改文章

1. 在 `docs/` 底下新增或修改 Markdown。
2. 若要出現在側邊導覽，更新 `mkdocs.yml` 的 `nav`。
3. 執行 `uv run mkdocs build --strict`，確認沒有缺頁、缺圖或 broken link。
4. 不要提交 `site/` 產物。

## 部署

GitHub Pages 由 GitHub Actions 建置與部署。推送到部署 workflow 指定分支後，CI 會使用 uv 同步環境，執行 strict build，再上傳 `site/` artifact 到 GitHub Pages。

Read the Docs 使用 `requirements.txt` 作為 pip fallback；本機開發與主要 CI 以 uv 為準。

## 疑難排解

如果 `uv` 指令找不到，重新開啟終端機，或確認 uv 安裝路徑已加入 `PATH`。

如果環境不同步，先刪除 `.venv` 後重新執行：

```bash
uv sync --locked
```

如果 strict build 失敗，先依錯誤訊息修正缺少的文件、圖片或錨點，再重新執行：

```bash
uv run mkdocs build --strict
```
