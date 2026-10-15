@echo off
chcp 65001 >nul
title 婴儿哭声识别后端（SSL 自训练 checkpoint 版）

echo ============================================
echo   婴儿哭声识别后端服务 · SSL 版
echo   Wav2Vec2-base 微调 · 6 类 · 端口 8800
echo ============================================
echo.
echo checkpoint : %BABY_CRY_CHECKPOINT%
echo   （默认为 checkpoint-1500，可用环境变量 BABY_CRY_CHECKPOINT 覆盖）
echo.
echo 正在启动服务...
echo.

cd /d "%~dp0backend-ssl"

set "PYTHON=python"
%PYTHON% --version >nul 2>nul
if errorlevel 1 set "PYTHON=py -3"

%PYTHON% server.py

echo.
echo 服务已停止。
pause
