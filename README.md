# SENTI

Sentiment Extraction from Natural signals in real-Time for Interpretation

Projeto de prototipo para NPS multimodal em tempo real, com foco em capturar sinais do sentimento do individuo durante a interacao.

## Objetivo

O projeto busca evoluir para um pipeline multimodal em tempo real que combine sinais visuais e de audio para apoiar uma leitura continua de experiencia.

No estado atual, o prototipo combina:

- emocao facial por video
- transcricao local do audio
- sentimento do texto transcrito com modelo Transformer em portugues
- emocao do tom de voz por heuristicas acustico-prosodicas com `openSMILE`

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
6. o `openSMILE` extrai medidas acustico-prosodicas do audio em janelas curtas
7. uma heuristica baseada em `pitch/F0`, `intensidade` e `duracao/ritmo` estima a emocao da voz
8. o modelo `SYAS1-PTBR` classifica o sentimento do texto transcrito
9. o video exibe a emocao facial, o sentimento do video, a transcricao, o sentimento do texto e a emocao da voz

Para manter a interface responsiva:

- a emocao facial eh inferida de forma assincrona, em thread separada
- a emocao visual nao eh recalculada em todo frame
- a emocao da voz eh inferida em thread separada a partir de janelas de audio
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
- `openSMILE`: extracao de features acustico-prosodicas da voz
- `SYAS1-PTBR`: sentimento textual em portugues com `transformers`

O sentimento atual do audio usa inferencia local com Transformer. O modelo eh baixado automaticamente para `models/SYAS1-PTBR/` na primeira execucao, caso ainda nao exista localmente.

Regras atuais para sentimento do audio:

- a analise parcial so comeca quando existem pelo menos 8 palavras no buffer
- depois da primeira analise, uma nova reavaliacao parcial acontece a cada 5 palavras novas
- apenas as 100 palavras mais recentes sao enviadas ao modelo de sentimento

### Voz

O tom de voz eh avaliado separadamente do sentimento textual. O `openSMILE` extrai features do conjunto `eGeMAPSv02`, e uma heuristica converte esses sinais em uma emocao candidata da voz:

- `Alegria`
- `Tristeza`
- `Raiva`
- `Medo`
- `Surpresa`
- `Nojo`
- `Neutra`

Base conceitual da heuristica:

- o `openSMILE` continua sendo tratado como um extrator de caracteristicas brutas
- no projeto, ele e usado para capturar medidas acustico-prosodicas como `frequencia fundamental/F0`, `intensidade/loudness` e `duracao`
- a heuristica foi documentada a partir de trabalhos que relacionam explicitamente essas medidas com as emocoes basicas em falantes do portugues brasileiro

Sinais usados pela heuristica:

- `F0 / pitch medio`
- `pico de pitch`
- `variabilidade de pitch`
- `loudness medio`
- `pico de loudness`
- `taxa de segmentos vozeados`
- `duracao media de segmentos vozeados`

Comportamento atual da heuristica:

- usa janela de audio de 2 segundos com atualizacao a cada 1 segundo
- precisa de algumas janelas iniciais para formar um baseline do proprio falante
- compara a janela atual contra esse baseline, em vez de usar limiares absolutos
- produz uma emocao de voz separada para uso futuro na fusao multimodal do sentimento
- quando a evidencia acustica eh fraca ou ambigua, o sistema prefere retornar `Indisponivel` em vez de forcar uma emocao
- `Neutra` eh usada quando o sinal existe, mas nao apresenta marcador emocional forte

Tabela de padroes acustico-prosodicos de referencia para a emocao da voz:

| Emocao | Duracao / velocidade de fala | Frequencia fundamental (F0 / pitch) | Intensidade (loudness / energia) |
| --- | --- | --- | --- |
| `Alegria` | menor duracao; fala mais acelerada e menor taxa de elocucao | maior; valores altos associados a uma valencia positiva | maior; apresenta alta intensidade media |
| `Tristeza` | nao discrimina isoladamente | menor; com menor media e menor variabilidade de F0 | menor; apresenta baixos valores medios e minimos de intensidade |
| `Raiva` | nao discrimina isoladamente | a media isolada discrimina pouco, mas ha alta energia geral | maior; apresenta picos maximos de intensidade |
| `Medo` | menor duracao; fala mais curta e com menor alongamento de segmentos | a media isolada discrimina pouco, mas ha a maior variabilidade de F0 | menor; baixa intensidade, proxima da tristeza |
| `Surpresa` | nao discrimina isoladamente | maior; com media de F0 alta e frequencia maxima elevada | nao discrimina isoladamente |
| `Nojo` | maior duracao; fala mais alongada e lentificada | nao discrimina isoladamente | nao discrimina isoladamente |
| `Neutra` | nao discrimina isoladamente | menor; valores baixos, proximos aos da tristeza | nao discrimina isoladamente |

Legenda da tabela:

- `maior`: parametro significativamente elevado em relacao ao padrao
- `menor`: parametro significativamente reduzido em relacao ao padrao
- `nao discrimina`: a variacao isolada desse parametro nao eh o principal fator para identificar a emocao

Principais heuristicas detalhadas para o algoritmo:

- `Nojo`: buscar anomalias de duracao. Na documentacao, o nojo se destaca principalmente pelos maiores alongamentos de fala.
- `Alegria` vs `Surpresa`: ambas podem apresentar `F0` alto. Para separa-las, observar `intensidade` e `duracao`: `Alegria` combina `F0` alto com alta intensidade media e fala acelerada; `Surpresa` tende a aparecer mais pelos picos absolutos de `F0`, sem o mesmo destaque de intensidade.
- `Raiva`: priorizar picos de intensidade. A raiva se destaca pelos maiores picos de energia vocal.
- `Medo`: combinar baixa intensidade com alta variabilidade de `F0` e fala encurtada.
- `Tristeza` vs `Neutra`: ambas podem ter `F0` baixo, mas `Tristeza` tende a reduzir tambem variabilidade de `F0` e intensidade geral.

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
- o `openSMILE` foi adicionado para capturar sinais paralinguisticos do tom de voz sem depender de modelo de audio pesado
- o sentimento textual do audio usa `SYAS1-PTBR` via `transformers`
- as inferencias mais pesadas foram desacopladas para reduzir travamentos da interface
- o overlay do texto usa fontes do Windows para texto e emoji
- o estado atual eh de desenvolvimento local com intencao edge, nao de deploy final em dispositivo offline

## Referencias

- Aguiar, A. C. de, Constantini, A. C., Moraes, R. M. de, & Almeida, A. A. "Medidas acustico-prosodicas discriminam as emocoes de falantes do portugues brasileiro". *CoDAS*, 2025.
- Eyben, F., Scherer, K., Schuller, B., Sundberg, J., Andre, E., Busso, C., Devillers, L., Epps, J., Laukka, P., Narayanan, S., & Truong, K. "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing".
- Jorge, Ana Cristina Aparecida. "Analise da percepcao da prosodia afetiva de pacientes com esquizofrenia". Tese de Doutorado, FFLCH-USP, 2023.
