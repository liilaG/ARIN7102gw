param(
  [string]$OutputDir = "dist\FinancialChatbot",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dist = Join-Path $Root $OutputDir

if (Test-Path $Dist) {
  Remove-Item $Dist -Recurse -Force
}
New-Item -ItemType Directory -Path $Dist | Out-Null

$dirs = @("query_intelligence", "scripts", "data", "models", "config")
foreach ($dir in $dirs) {
  Copy-Item (Join-Path $Root $dir) (Join-Path $Dist $dir) -Recurse -Force
}

$files = @("requirements.txt", "start.bat", ".env.example", "README_CN.md")
foreach ($file in $files) {
  Copy-Item (Join-Path $Root $file) (Join-Path $Dist $file) -Force
}

& $Python -m venv (Join-Path $Dist ".venv")
$VenvPython = Join-Path $Dist ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Dist "requirements.txt")

$readme = @"
Financial Chatbot by Group 4.2 启动说明
=====================================

1. 打开 config\app_config.json。
2. 把 deepseek.api_key 改成你的 DeepSeek API Key。
3. 如需修改 API 地址、模型、页面标题、输入框提示或按钮文字，也在 config\app_config.json 中修改。
4. 双击 start.bat，程序会启动本地服务并自动打开浏览器。
5. 默认地址：http://127.0.0.1:8765/

说明：
- live_data.enabled 默认为 true，会优先尝试联网数据源；失败时仍会使用本地数据。
- 也可以通过 .env 或系统环境变量覆盖配置，例如 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL、CHATBOT_TITLE。
- 不要把真实 API Key 提交到代码仓库。
"@

$readme | Set-Content -Path (Join-Path $Dist "README_启动说明.txt") -Encoding UTF8
Write-Host "Built Windows distribution at $Dist"
