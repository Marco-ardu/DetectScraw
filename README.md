# DetectScraw
## venv
```shell
virtualenv venv
```

## dependencies
```shell
pip install -r requirements.txt
```

## .ui->.py
```shell
pyuic5 ui/main.ui -o ui/ui_main.py
```

## RUN
```
python Program.py
```

## pyinstaller
```shell
pyinstaller -i cam.ico Program.py --add-data="cam.ico;." --add-data="sound\\*;sound\\" --add-data="config.yml;." --add-data="cameraFunc\models\yolox_nano_components_openvino_2021.4_6shave.blob;cameraFunc\\models\\" --distpath="C:\\OAK_Detection_Scraw" --exclude-module=pyinstaller -y -w 
```