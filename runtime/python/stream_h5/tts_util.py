import io
import math
# from tools.audio.np import float_to_int16
import os
import time
import wave

import numpy as np
from numba import jit


@jit
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def getTime():
    # 获取当前时间戳（秒数）
    timestamp = time.time()
    # 将秒数转换为毫秒数，并去除小数部分
    milliseconds = round(timestamp * 1000)
    return milliseconds


def getDir(username, session_hash, run):
    return f'media/{username}/{session_hash}/{run}'


'''
处理wav字节数据
'''


def processBinaryWav(binary_data, isfirstchunk=False):
    # 从第一个块标头中剥离长度信息，从后续块中完全删除标头，这样audio才能正常播放wav字节流
    if isfirstchunk:
        binary_data = (
                binary_data[:4] + b"\xff\xff\xff\xff" + binary_data[8:]
        )
        binary_data = (
                binary_data[:40] + b"\xff\xff\xff\xff" + binary_data[44:]
        )
    else:
        binary_data = binary_data[44:]
    return binary_data


def initStreamDict(root: dict, fields):
    username = fields['username']
    session_hash = fields['session_hash']
    run = fields['run']

    if root.get(username) is None:
        root[username] = {}

    if root[username].get(session_hash) is None:
        root[username][session_hash] = {}

    if root[username][session_hash].get(run) is None:
        root[username][session_hash][run] = []

    return root


def processWavs(wavs, params, callback):
    username = params['username']
    session_hash = params['session_hash']
    run = params['run']

    dir = getDir(username, session_hash, run)

    for i, wav in enumerate(wavs):

        filename = f'{i}.wav'
        fp = f'{dir}/{filename}'

        if not os.path.exists(dir):
            os.makedirs(dir)

        buf = io.BytesIO()
        with wave.open(fp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(float_to_int16(wav))
        buf.seek(0, 0)

        result = {
            "username": username,
            "session_hash": session_hash,
            "run": run
        }
        if (i == 0):
            result["url"] = f'stream/{username}/{session_hash}/{run}'
            result["isOpen"] = True
        callback(result)

    # 添加结束消息
    callback({
        "username": username,
        "session_hash": session_hash,
        "run": run,
        "isClosed": True
    })


def fetchWav(filePath: str, isFirstChunk: bool):
    with wave.open(filePath, 'rb') as wav:
        sample_rate = wav.getframerate()
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        nframes = wav.getnframes()
        audio_data = wav.readframes(nframes)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        buf.seek(0, 0)
        audio_data = buf.getvalue()
    return processBinaryWav(audio_data, isFirstChunk)
