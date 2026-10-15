@echo off
chcp 65001 >nul
title 婴儿哭声 Demo 一键启动（8800 后端 + 静态页 + 浏览器）
setlocal enabledelayedexpansion

REM ============== 配置 ==============
set "PYTHON=python"
%PYTHON% --version >nul 2>nul
if errorlevel 1 set "PYTHON=py -3"

set "BACKEND_DIR=%~dp0backend-ssl"
set "DEMO_DIR=%~dp0demo"
set "STATIC_PORT=5500"
set "BACKEND_PORT=8800"
set "BROWSER_URL=http://127.0.0.1:%STATIC_PORT%/index.html"

echo ============================================
echo   婴儿哭声 Demo 一键启动
echo   后端 : http://127.0.0.1:%BACKEND_PORT%  (SSL 自训练 Wav2Vec2)
echo   Demo : http://127.0.0.1:%STATIC_PORT%
echo ============================================
echo.

if not exist "%DEMO_DIR%\index.html" (
  echo [错误] 找不到 demo 文件: %DEMO_DIR%\index.html
  echo 请确认 demo 目录存在，或修改本脚本 DEMO_DIR。
  pause & exit /b 1
)

REM ---- 1. 检查 8800 后端，未起则在新窗口启动 ----
echo [1/3] 检查 %BACKEND_PORT% ...
netstat -ano | findstr ":%BACKEND_PORT% " | findstr LISTENING >nul
if %errorlevel%==0 (
  echo   后端已在运行，跳过启动。
) else (
  echo   未检测到后端，启动 server.py（含 CORS）...
  start "baby-cry-backend-ssl" cmd /k "cd /d %BACKEND_DIR% && set PORT=%BACKEND_PORT% && %PYTHON% server.py"
)

REM ---- 2. 启动本地静态服务器（后台窗口）----
echo [2/3] 启动静态服务器 :%STATIC_PORT% ...
start "baby-cry-demo-static" /min cmd /k "cd /d %DEMO_DIR% && set PORT=%STATIC_PORT% && %PYTHON% -m http.server %STATIC_PORT% --bind 127.0.0.1"

REM ---- 3. 等静态服务器起来后打开浏览器 ----
echo [3/3] 打开浏览器 ...
timeout /t 3 >nul
start "" "%BROWSER_URL%"

echo.
echo 完成。后端(8800)与静态页(5500)已在各自窗口运行。
echo 关闭本脚本不会停止后端/静态窗口；在各自窗口按 Ctrl+C 停止。
pause
endlocal
