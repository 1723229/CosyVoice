# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
import sys

logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import time
import random
import runtime.python.stream_h5.tts_stream as tts_stream
from pydantic import BaseModel

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


class GenerateJoinRequest(BaseModel):
    username: str
    session_hash: str
    input: str


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.post("/stream/queue/join")
async def streamQueueJoin(data: GenerateJoinRequest):
    return tts_stream.streamQueueJoin(data, cosyvoice)


@app.get("/stream/queue/data")
async def streamQueueData(username: str, session_hash: str, run: str):
    return tts_stream.streamQueueData(username, session_hash, run)


@app.get("/stream/{username}/{session_hash}/{run}")
async def streamAudio(username, session_hash, run):
    return tts_stream.streamAudio(username, session_hash, run)


@app.post("/inference")
async def inference(tts_text: str = Form(), stream: bool = Form()):
    prompt_wav = "../../../zero_shot_kf_prompt.wav"
    prompt_text = "近年来，随着深度学习技术的飞速发展，自然语言处理领域取得了显著的进步。"
    prompt_speech_16k = load_wav(prompt_wav, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream)
    if stream:
        return StreamingResponse(generate_data(model_output))
    else:
        tts_speeches = []
        for model_output in model_output:
            tts_speeches.append(model_output['tts_speech'])
        tts_speeches = torch.concat(tts_speeches, dim=1)
        wav_name = str(round(time.time() * 1000)) + str(random.randint(100000, 999999)) + ".wav"
        torchaudio.save('{}/{}'.format("/opt/tts_file", wav_name), tts_speeches, sample_rate=22050)
        return StreamingResponse(wav_name)


@app.get("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
