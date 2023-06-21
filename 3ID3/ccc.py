#microphone
from machine import Pin
from machine import I2S

bck_pin = Pin(23)
ws_pin = Pin(5)
sdin_pin = Pin(22)
audio_in = I2S(0,
               sck=bck_pin, ws=ws_pin, sd=sdin_pin,
               mode=I2S.RX,
               bits=16,
               format=I2S.MONO,
               rate=8000,
               ibuf=64000)

ibuf = bytearray(64000)
wav = open('audio.wav', 'wb')

# Write the WAV header
wav.write(b'RIFF')
wav.write(int.to_bytes(36 + 64000, 4, 'little'))
wav.write(b'WAVE')
wav.write(b'fmt ')
wav.write(int.to_bytes(16, 4, 'little'))
wav.write(int.to_bytes(1, 2, 'little'))
wav.write(int.to_bytes(1, 2, 'little'))
wav.write(int.to_bytes(8000, 4, 'little'))
wav.write(int.to_bytes(16000, 4, 'little'))
wav.write(int.to_bytes(2, 2, 'little'))
wav.write(int.to_bytes(16, 2, 'little'))
wav.write(b'data')
wav.write(int.to_bytes(64000, 4, 'little'))

print('Starting')

num_read = audio_in.readinto(ibuf)
wav.write(ibuf)
wav.close()
audio_in.deinit()

print('Done')



# oled
from machine import Pin, SoftI2C
from time import sleep
import ssd1306

# 创建i2c对象
i2c = SoftI2C(scl=Pin(19), sda=Pin(21))

# 宽度高度
oled_width = 128
oled_height = 64

# 创建oled屏幕对象
oled = ssd1306.SSD1306_I2C(oled_width, oled_height, i2c)

# 在指定位置处显示文字
oled.text('你好!', 0, 0)
oled.text('Hello, World 2!', 0, 10)
oled.text('Hello, World 3!', 0, 20)

oled.show()

#wifi
import network
import machine
import time

pin2 = machine.Pin(2, machine.Pin.OUT)
wlan = network.WLAN(network.STA_IF)  # create station interface
wlan.active(True)  # activate the interface
wlan.scan()  # scan for access points

# 联网
while True:
    pin2.value(1)
    if wlan.isconnected():
        pin2.value(0)
        break
    else:
        try:
            wlan.connect('liubf', '200112290')  # connect to an AP
            time.sleep(1)
            pin2.value(0)
        except:
            pin2.value(0)
            time.sleep(1)
            continue
print(wlan.config('mac'))  # get the interface's MAC address
print(wlan.ifconfig())  # get the interface's IP/netmask/gw/DNS addresses

# 麦克风INMP441收集


# 百度API，语音识别
import urequests as requests

with open('audio.wav', 'rb') as f:
    f.seek(44)  # 跳过文件头
    ibuf = f.read()
# 假设 ibuf 是一个包含音频数据的字节数组
url = "https://vop.baidu.com/server_api?cuid={}&token={}".format('D8-BB-C1-E1-8E-8F',
                                                                 '24.461acd56fe26e4efcf1a8d5a6a61712f.2592000.1687446530.282335-33894227')
headers = {
    'Content-Type': 'audio/wav;rate=16000'
}

response = requests.post(url, headers=headers, data=ibuf)

print(response.text)

# 发送request,OPENAI
import urequests

url = 'http://8.130.68.97:3001/web_fourth'
headers = {'Content-Type': 'application/json'}

try:
    response = urequests.get(url, headers=headers)
    print(response.text)
except:
    print('requests error')

wlan.disconnect()

#yinxiang
from machine import I2S
from machine import Pin

"""
GPIO15 -- DIN
GPIO2 --- BCLK
GPIO4 -- LRC
GND -- GND
5V或3.3V -- VCC
"""

# 初始化引脚定义
sck_pin = Pin(2)  # 串行时钟输出
ws_pin = Pin(4)  # 字时钟
sd_pin = Pin(15)  # 串行数据输出

"""
sck 是串行时钟线的引脚对象
ws 是单词选择行的引脚对象
sd 是串行数据线的引脚对象
mode 指定接收或发送
bits 指定样本大小（位），16 或 32
format 指定通道格式，STEREO（左右声道） 或 MONO(单声道)
rate 指定音频采样率（样本/秒）
ibuf 指定内部缓冲区长度（字节）
"""

# 初始化i2s
audio_out = I2S(1, sck=sck_pin, ws=ws_pin, sd=sd_pin, mode=I2S.TX, bits=16, format=I2S.MONO, rate=16000, ibuf=20000)

wavtempfile = "audio.wav"
with open(wavtempfile, 'rb') as f:
    # 跳过文件的开头的44个字节，直到数据段的第1个字节
    pos = f.seek(44)

    # 用于减少while循环中堆分配的内存视图
    wav_samples = bytearray(1024)
    wav_samples_mv = memoryview(wav_samples)

    print("开始播放音频...")

    # 并将其写入I2S DAC
    while True:
        try:
            num_read = f.readinto(wav_samples_mv)

            # WAV文件结束
            if num_read == 0:
                break

            # 直到所有样本都写入I2S外围设备
            num_written = 0
            while num_written < num_read:
                num_written += audio_out.write(wav_samples_mv[num_written:num_read])

        except Exception as ret:
            print("产生异常...", ret)
            break



