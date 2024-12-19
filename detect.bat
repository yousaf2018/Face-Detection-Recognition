set CONDAPATH=D:\Miniconda3
set ENVNAME=face-recognition

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

python source_code/testing/inference.py

