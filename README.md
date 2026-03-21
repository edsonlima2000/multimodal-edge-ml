# SENTI

Sentiment Extraction from Natural signals in real-Time for Interpretation

Projeto de prototipo para NPS multimodal em tempo real, com foco em capturar sinais do sentimento do individuo durante a interacao.

## Objetivo

O projeto busca evoluir para um pipeline multimodal em tempo real que combine sinais visuais e de audio para apoiar uma leitura continua de experiencia.

No estado atual, o prototipo combina:

- emocao facial por video
- transcricao local do audio
- sentimento do texto transcrito com modelo Transformer em portugues

## Edge

Este projeto ainda nao esta empacotado nem otimizado para um dispositivo edge offline dedicado, mas ja segue uma direcao edge-aware.

O que eh verdade hoje:

- a inferencia acontece localmente, sem depender de API externa
- o video eh capturado pela webcam local
- o audio eh capturado pelo microfone local
- a deteccao facial, a transcricao e a classificacao atual rodam na propria maquina

O que ainda nao afirmamos:

- que o projeto ja esta pronto para Raspberry Pi, Jetson, Android ou outro hardware restrito
- que o pipeline atual ja esta otimizado em latencia, memoria ou consumo
- que todos os componentes atuais sao os mais leves possiveis

Direcao de desenvolvimento:

- cada novo componente deve ser avaliado tambem pelo peso computacional e pela facilidade de portar para cenarios mais lite
- quando houver alternativa equivalente, a preferencia eh por componentes mais leves
- a estrategia atual privilegia processamento local, streaming e baixo acoplamento entre modalidades

## Pipeline atual

O fluxo principal do prototipo multimodal funciona assim:

1. a webcam captura o video com OpenCV
2. o MediaPipe detecta o rosto em cada frame
3. o recorte facial em escala de cinza eh enviado para o modelo MiniXception
4. o microfone captura audio em streaming
5. o Vosk transcreve a fala localmente em portugues
6. o modelo `SYAS1-PTBR` classifica o sentimento do texto transcrito
7. o video exibe a emocao facial, o sentimento do video, a transcricao e o sentimento do audio

Para manter a interface responsiva:

- a emocao facial eh inferida de forma assincrona, em thread separada
- a emocao visual nao eh recalculada em todo frame
- o sentimento do audio usa janela deslizante maxima de 100 palavras
- a reavaliacao parcial do sentimento do audio acontece por progresso textual, e nao a cada mudanca curta de transcricao

## Componentes

### Video

- `MediaPipe Face Detector`: deteccao facial em tempo real
- `MiniXception`: classificacao de emocao facial

O modelo `MiniXception` usado neste projeto foi treinado no dataset `FER-2013`.

Emocoes exibidas no video:

- 😠 Raiva (Angry)
- 🤢 Nojo (Disgust)
- 😨 Medo (Fear)
- 😊 Feliz (Happy)
- 😢 Triste (Sad)
- 😲 Surpreso (Surprise)
- 😐 Neutro (Neutral)

O script tambem deriva um sentimento simplificado para o video:

- `Positivo`
- `Negativo`
- `Neutro`

### Audio

- `sounddevice`: captura de audio do microfone
- `Vosk`: transcricao offline em streaming
- `SYAS1-PTBR`: sentimento textual em portugues com `transformers`

O sentimento atual do audio usa inferencia local com Transformer. O modelo eh baixado automaticamente para `models/SYAS1-PTBR/` na primeira execucao, caso ainda nao exista localmente.

Regras atuais para sentimento do audio:

- a analise parcial so comeca quando existem pelo menos 8 palavras no buffer
- depois da primeira analise, uma nova reavaliacao parcial acontece a cada 5 palavras novas
- apenas as 100 palavras mais recentes sao enviadas ao modelo de sentimento

## Estrutura

- `hello_mediapipe.py`: baseline visual com MediaPipe + MiniXception
- `senti.py`: prototipo multimodal com video, audio, transcricao e sentimento
- `models/blaze_face_short_range.tflite`: detector de rosto usado pelo MediaPipe
- `models/fer2013_mini_XCEPTION.102-0.66.hdf5`: modelo de emocao facial
- `models/vosk-model-small-pt-0.3/`: modelo pequeno de ASR em portugues para o Vosk
- `models/SYAS1-PTBR/`: modelo de sentimento textual em portugues
- `requirements.txt`: dependencias diretas do projeto

## Requisitos

- Windows
- Python 3.12
- webcam conectada
- microfone funcional

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
pip install -r requirements.txt
```

## Execucao

Rodar o prototipo multimodal:

```powershell
python senti.py
```

Rodar somente a baseline visual:

```powershell
python hello_mediapipe.py
```

Para sair da janela do video, pressione `q`.

## Observacoes

- o script usa `MediaPipe Tasks`, nao a API antiga `mp.solutions`
- o Vosk foi escolhido nesta fase por ser offline, leve e orientado a streaming
- o sentimento textual do audio usa `SYAS1-PTBR` via `transformers`
- as inferencias mais pesadas foram desacopladas para reduzir travamentos da interface
- o overlay do texto usa fontes do Windows para texto e emoji
- o estado atual eh de desenvolvimento local com intencao edge, nao de deploy final em dispositivo offline
