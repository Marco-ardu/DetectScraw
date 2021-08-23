@REM GOTO D:\qtCart\carEnv\Scripts\activate.bat
@REM python Program.py
@echo off
cmd /k "cd /d D:\qtCart\carEnv\Scripts\ & activate & cd /d    D:\qtCart\ & python Program.py"
PAUSE
call carEnv\Scripts\activate.bat
python -m Program.py