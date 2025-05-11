@echo off

REM --- 配置 (请务必修改这些值) ---
REM 1. 设置你的 GitHub 仓库 URL
set GITHUB_REPO_URL=https://github.com/WU-rxj/machine_learning_program.git

REM 2. 设置你的默认分支名称 (通常是 main 或 master)
set BRANCH_NAME=main

REM 3. 设置默认提交信息 (如果留空，脚本会使用 "Auto-commit by script")
set DEFAULT_COMMIT_MESSAGE=Auto-commit by script

REM --- 检查 Git 是否在仓库中 ---
if not exist .git (
    echo 当前目录不是一个 Git 仓库。正在初始化...
    git init
    if %errorlevel% neq 0 (
        echo Git 仓库初始化失败。
        pause
        exit /b 1
    )
    echo Git 仓库已初始化。
)

REM --- 获取提交信息 (如果用户在命令行提供了参数，则使用参数作为提交信息) ---
set COMMIT_MESSAGE=%~1
if "%COMMIT_MESSAGE%"=="" (
    set COMMIT_MESSAGE=%DEFAULT_COMMIT_MESSAGE%
)

REM --- Git 命令 ---
echo.
echo 正在添加所有文件到暂存区...
git add .

echo.
echo 正在提交更改 (信息: "%COMMIT_MESSAGE%")...
git commit -m "%COMMIT_MESSAGE%"

REM --- 检查并设置远程仓库 ---
git remote -v | findstr /C:"origin	%GITHUB_REPO_URL% (push)" >nul
if %errorlevel% neq 0 (
    echo 正在设置/更新远程 'origin' 为: %GITHUB_REPO_URL%
    git remote rm origin >nul 2>nul
    git remote add origin %GITHUB_REPO_URL%
    if %errorlevel% neq 0 (
        echo 设置远程仓库失败。
        pause
        exit /b 1
    )
)

REM --- 确保在正确的分支上 ---
git branch --show-current | findstr /C:"%BRANCH_NAME%" >nul
if %errorlevel% neq 0 (
    echo 正在切换到/创建 '%BRANCH_NAME%' 分支...
    git checkout -B %BRANCH_NAME%
    if %errorlevel% neq 0 (
        echo 切换/创建分支 '%BRANCH_NAME%' 失败。
        pause
        exit /b 1
    )
)

REM --- 推送到 GitHub ---
echo.
echo 正在推送到 GitHub 仓库 (%BRANCH_NAME% 分支)...
git push -u origin %BRANCH_NAME%

if %errorlevel% equ 0 (
    echo.
    echo 代码已成功上传到 GitHub!
) else (
    echo.
    echo 上传过程中发生错误。请检查错误信息并重试。
)

echo.
pause