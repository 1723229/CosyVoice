'''
Copyright Longing  2024. All Rights Reserved. 
Author: 周腾蛟
Date: 2024-08-15 17:44:17
LastEditors: 周腾蛟
LastEditTime: 2024-08-29 17:34:30
FilePath: /CosyVoice/openai.py
Description: 流式语音合成
Other: 下面是变更记录，请主动填写。
Change Log:
  <author>      <time>       <version>     <description>
   周腾蛟      2024-08-15      0.0.1          create
'''
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from fastapi.responses import StreamingResponse
from fastapi import FastAPI
import asyncio
import runtime.python.stream_h5.tts_util as ttsutil
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

cosyvoice = CosyVoice("/data/models/CosyVoice-300M-25Hz")
prompt_speech_16k = load_wav('../../..//zero_shot_kf_prompt.wav', 16000)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，也可以指定具体的源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

streamDict = {}
array = []


class GenerateJoinRequest(BaseModel):
    username: str
    session_hash: str
    input: str


thread_pool = ThreadPoolExecutor()


# 发起tts请求
def streamQueueJoin(data: GenerateJoinRequest):
    username = data.username
    session_hash = data.session_hash
    run = ttsutil.getTime()
    wavs = InferCosyVoice(input=data.input, stream=True)

    param = {'username': username, 'session_hash': session_hash, 'run': f'{run}'}

    ttsutil.initStreamDict(streamDict, param)

    # 声明每段音频处理 完后的闭包
    # 添加到对应会话消息队列中
    def callback(params):
        username = params['username']
        session_hash = params['session_hash']
        run = params['run']
        array = streamDict[username][session_hash][run]
        array.append(params)

    thread_pool.submit(ttsutil.processWavs, wavs, param, callback)

    return {
        "username": username,
        "session_hash": session_hash,
        "run": run
    }

    # 消息推送，每生成一段音频便推送一次消息


def streamQueueData(username: str, session_hash: str, run: str):
    index = 0

    async def generate_stream():
        nonlocal index
        array = streamDict[username][session_hash][run]
        while (True):
            if (len(array) and index < len(array)):
                v = array[index]
                yield f'data:{json.dumps(v)}\n\n'
                index += 1
                if (v.get('isClosed') == True):
                    break
            await asyncio.sleep(0.001)

    yoyo = generate_stream()

    return StreamingResponse(yoyo, media_type="text/event-stream")


    # 获取音频流


def streamAudio(username, session_hash, run):
    dir = ttsutil.getDir(username, session_hash, run)
    index = 0

    try:
        array = streamDict[username][session_hash][run]
    except:
        array = None
        pass

    async def generate_stream():
        nonlocal index
        if (array != None):
            # 如果流式缓存中有临时记录，说明正在生成，否则是历史
            while (True):
                if (len(array) and index < len(array)):
                    v = array[index]
                    if (v.get('isClosed') == True):
                        print('播放完成')
                        streamDict[username][session_hash][run] = None
                        break
                    yield ttsutil.fetchWav(f'{dir}/{index}.wav', index == 0)
                    index += 1
                await asyncio.sleep(0.001)

        else:
            while (True):
                try:
                    yield ttsutil.fetchWav(f'{dir}/{index}.wav', index == 0)
                    index += 1
                except IOError as e:
                    print('读取不到文件，请求结束', e)
                    break

    yoyo = generate_stream()

    return StreamingResponse(yoyo, media_type="audio/wav")


def InferCosyVoice(input: str, stream: bool):
    wavs = cosyvoice.inference_zero_shot(
        input,
        '近年来，随着深度学习技术的飞速发展，自然语言处理领域取得了显著的进步。',
        prompt_speech_16k,
        stream=stream)
    # 返回wav流
    for wav in wavs:
        yield wav['tts_speech'].numpy()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=45000)
