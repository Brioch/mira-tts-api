# mira-tts-api
[MiraTTS](https://huggingface.co/YatharthS/MiraTTS) is a finetune of the excellent [Spark-TTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) model for enhanced realism and stability performing on par with closed source models. 
This repository also heavily optimizes Mira with [Lmdeploy](https://github.com/InternLM/lmdeploy) and boosts quality by using [FlashSR](https://github.com/ysharma3501/FlashSR) to generate high quality audio at over **100x** realtime!

This repository adds a simple API to generate audio from text using MiraTTS. It is compatible with the OpenAI API format and can be used with any OpenAI compatible client.

## Key benefits
- Incredibly fast: Over 100x realtime by using Lmdeploy and batching.
- High quality: Generates clear and crisp 48khz audio outputs which is much higher quality then most models.
- Memory efficient: Works within 6gb vram.
- Low latency: Latency can be low as 100ms.

## Install

### Using uv

`uv` is supported as a faster alternative to conda and pip.

```sh
uv sync
uv run main.py
```

## Usage

```sh
uv run main.py
```

Server will run by default on http://127.0.0.1:5001/v1/audio/speech.

Parameters are set with environment variables.

Copy `.env.dist` to `.env` and edit it to your needs.

```sh
cp .env.dist .env
```

Voice are expected to be wav audio files. For example, with a directory `/home/user/voices` containing `alloy.wav`, you would run the server with the following env vars:

- `SUPPORTED_VOICES=alloy`
- `VOICES_DIR=/home/user/voices`

### Environment Variables

```
API_HOST              Host to run the server on. Default: 0.0.0.0
API_PORT              Port to run the server on. Default: 5001
VOICES_DIR            Path to the audio prompt files dir. Default: ./voices
SUPPORTED_VOICES      Comma-separated list of supported voices. Example: 'alloy,ash'. Default is empty so all voices in the voices dir are loaded.
CORS_ALLOW_ORIGIN     CORS allowed origin. Default: *
SEED                  Seed for reproducibility. Default: 0 (random)
WEB_PORT              Port to run the web UI on when using the Dockerfile. Default: 8080
```

### Using the API

## /v1/audio/speech

See [OpenAI Compatible API Speech endpoint](https://platform.openai.com/docs/api-reference/audio/createSpeech). This API takes a json containing an input text and a voice and replies with the TTS audio data. 

Example API call with `curl`:

```sh
curl -X POST http://localhost:5001/v1/audio/speech -H "Content-Type: application/json" -d '{"input": "Hello, this is a test.", "voice": "alloy"}' --output speech.wav
```

## /tts

This API is similar to the OpenAI API but it allows for more parameters.

Parameters are text, predefined_voice_id, model, speed_factor, cfg_weight, temperature, exaggeration, output_format, seed, language_id.

```sh
curl -X POST http://localhost:5001/tts -H "Content-Type: application/json" -d '{"text": "Hello, this is a test.", "predefined_voice_id": "alloy"}' --output speech.wav
```

### Using the web UI

First, run the API server. Then start the web UI server:

```sh
python -m http.server 8080 -d public -b 127.0.0.1
```

Then open `http://localhost:8080` in your browser. You can set the API URL, enter text and select a voice to generate speech.

### Usage in SillyTavern

You can use the "Chatterbox" or "OpenAI compatible" TTS options in SillyTavern to use this API.
