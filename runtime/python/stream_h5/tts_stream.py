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

from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from fastapi.responses import StreamingResponse
import asyncio
import runtime.python.stream_h5.tts_util as ttsutil
from cosyvoice.utils.file_utils import load_wav

prompt_speech_16k = load_wav('../../..//zero_shot_kf_prompt.wav', 16000)

streamDict = {}

class GenerateJoinRequest(BaseModel):
    username: str
    session_hash: str
    input: str


thread_pool = ThreadPoolExecutor()


def stream_queue_join(data: GenerateJoinRequest, cosyvoice):
    username = data.username
    session_hash = data.session_hash
    run = ttsutil.getTime()
    output_wav = InferCosyVoice(cosyvoice=cosyvoice, tts_text=data.input, stream=True)

    param = {'username': username, 'session_hash': session_hash, 'run': f'{run}'}

    ttsutil.initStreamDict(streamDict, param)

    # 声明每段音频处理 完后的闭包
    # 添加到对应会话消息队列中
    def callback(params):
        user_name = params['username']
        session_hash = params['session_hash']
        run = params['run']
        data_dict = streamDict[user_name][session_hash][run]
        data_dict.append(params)

    thread_pool.submit(ttsutil.processWavs, output_wav, param, callback)

    return {
        "username": username,
        "session_hash": session_hash,
        "run": run
    }


def stream_queue_data(username: str, session_hash: str, run: str):
    index = 0

    async def generate_stream():
        nonlocal index
        data_dict = streamDict[username][session_hash][run]
        while True:
            if len(data_dict) and index < len(data_dict):
                v = data_dict[index]
                yield f'data:{json.dumps(v)}\n\n'
                index += 1
                if v.get('isClosed'):
                    break
            await asyncio.sleep(0.001)

    yoyo = generate_stream()

    return StreamingResponse(yoyo, media_type="text/event-stream")


def stream_audio(username, session_hash, run):
    data_dir = ttsutil.getDir(username, session_hash, run)
    index = 0
    try:
        data_dict = streamDict[username][session_hash][run]
    except:
        data_dict = None
        pass

    async def generate_stream():
        nonlocal index
        if data_dict is not None:
            # 如果流式缓存中有临时记录，说明正在生成，否则是历史
            while True:
                if len(data_dict) and index < len(data_dict):
                    v = data_dict[index]
                    if v.get('isClosed'):
                        print('播放完成')
                        streamDict[username][session_hash][run] = None
                        break
                    yield ttsutil.fetchWav(f'{data_dir}/{index}.wav', index == 0)
                    index += 1
                await asyncio.sleep(0.001)

        else:
            while True:
                try:
                    yield ttsutil.fetchWav(f'{data_dir}/{index}.wav', index == 0)
                    index += 1
                except IOError as e:
                    print('读取不到文件，请求结束', e)
                    break

    yoyo = generate_stream()

    return StreamingResponse(yoyo, media_type="audio/wav")


def InferCosyVoice(cosyvoice, tts_text: str, stream: bool):
    output_wav = cosyvoice.inference_zero_shot(
        tts_text,
        '近年来，随着深度学习技术的飞速发展，自然语言处理领域取得了显著的进步。',
        prompt_speech_16k,
        stream=stream)
    # 返回wav流
    for wav in output_wav:
        yield wav['tts_speech'].numpy()
