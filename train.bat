set CONDAPATH=D:\Miniconda3
set ENVNAME=face-recognition

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

python source_code/training/train.py --img 640 --batch 16 --epochs 10 --data source_code/training/data/custom.yaml --weights source_code/training/yolov5s.pt --nosave --cache

