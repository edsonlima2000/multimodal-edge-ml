# multimodal-edge-ml

Projeto de teste para deteccao facial em tempo real com MediaPipe e classificacao de emocao com MiniXception.

## Edge

Este projeto ainda nao esta empacotado nem otimizado para um dispositivo edge offline dedicado.

O que eh verdade hoje:

- a inferencia acontece localmente, sem depender de API externa
- o video eh capturado pela webcam local
- a deteccao facial e a classificacao de emocao rodam na propria maquina

O que ainda nao afirmamos:

- que o projeto ja esta pronto para Raspberry Pi, Jetson, Android ou outro hardware restrito
- que o pipeline atual ja esta otimizado em latencia, memoria ou consumo
- que todos os componentes atuais sao os mais leves possiveis

Direcao de desenvolvimento:

- a intencao do projeto eh evoluir com foco em edge
- cada novo componente deve ser avaliado tambem pelo peso computacional e pela facilidade de uso futuro em cenarios mais lite
- quando houver alternativa equivalente, a preferencia eh por componentes mais leves e mais faceis de portar

## Visao geral

O fluxo atual do projeto funciona assim:

1. a webcam captura o video com OpenCV
2. o MediaPipe detecta o rosto em cada frame
3. o recorte facial em escala de cinza eh enviado para o modelo MiniXception
4. a emocao prevista eh desenhada no video com texto em portugues/ingles

O modelo `MiniXception` usado neste projeto foi treinado no dataset `FER-2013`.

Emocoes exibidas no video:

- 😠 Raiva (Angry)
- 🤢 Nojo (Disgust)
- 😨 Medo (Fear)
- 😊 Feliz (Happy)
- 😢 Triste (Sad)
- 😲 Surpreso (Surprise)
- 😐 Neutro (Neutral)

## Estrutura

- `hello_mediapipe.py`: script principal
- `models/blaze_face_short_range.tflite`: detector de rosto usado pelo MediaPipe
- `models/fer2013_mini_XCEPTION.102-0.66.hdf5`: modelo de emocao

## Requisitos

- Windows
- Python 3.12
- webcam conectada

Dependencias principais:

- `mediapipe`
- `opencv-python`
- `tensorflow`
- `pillow`
- `numpy`

## Ambiente virtual

Criar o ambiente:

```powershell
python -m venv .venv
```

Ativar no PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
pip install mediapipe opencv-python tensorflow pillow numpy
```

## Execucao

Com o ambiente ativado:

```powershell
python hello_mediapipe.py
```

Para sair da janela do video, pressione `q`.

## Observacoes

- o script usa `MediaPipe Tasks`, nao a API antiga `mp.solutions`
- o overlay do texto usa a fonte de emoji do Windows em `C:\Windows\Fonts\seguiemj.ttf`
- os modelos necessarios ja estao versionados na pasta `models/`
- o estado atual eh de desenvolvimento local com intencao edge, nao de deploy final em dispositivo offline
