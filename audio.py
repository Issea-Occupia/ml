import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, convolve2d,istft
import soundfile as sf

# =============== 1) 读音频 ===============
audio_path = "/home/isseaoccupia/ml/Heinrixh - 18 - Crawk - Awkward Exchange.mp3"   # 改这里喵～
x, sr = sf.read(audio_path)     # x: (T,) or (T, C)
# 双声道取单声道
if x.ndim == 2:
    x = x[:, 0]

# 归一化
x = x.astype(np.float32)
x /= (np.max(np.abs(x)) + 1e-9)

# =============== 2) 做“音频图片”：log 频谱 ===============
# STFT 参数：25ms窗，10ms hop（对语音/环境音很常用）
win = int(0.025 * sr)
hop = int(0.010 * sr)
noverlap = win - hop

n_fft = 1
while n_fft < win:
    n_fft *= 2


f, tt, Z = stft(x, fs=sr, nperseg=win, noverlap=noverlap, nfft=n_fft, boundary=None)
S = np.abs(Z)                      # 幅度谱
S_log = np.log(S + 1e-6)           # log 更直观（动态范围压缩）
# S_log 形状： (freq_bins, time_frames) —— 就是“矩阵化”喵！

# =============== 3) 定义方向性 kernel（和你的一样风格） ===============
kernel_v = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

kernel_h = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)

# =============== 4) 2D 卷积：提取“纵/横变化” ===============
# 对谱图做卷积：注意这里的“纵向/横向”指的是谱图坐标
# 纵轴 = 频率，横轴 = 时间
out_v = convolve2d(S_log, kernel_v, mode="same", boundary="symm")
out_h = convolve2d(S_log, kernel_h, mode="same", boundary="symm")

# =============== 5) 归一化显示（仿照你 maxabs 的写法） ===============
def norm_signed(a):
    return a / (np.max(np.abs(a)) + 1e-8)

S_show = norm_signed(S_log)  # 只是为了把原谱图对比好看点（也可以不归一化）
nv = norm_signed(out_v)
nh = norm_signed(out_h)

# =============== 6) 画图：仿照你 1 行 3 列的布局 ===============
fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

axs[0].imshow(S_log, aspect="auto", origin="lower", cmap="gray")
axs[0].set_title("Original (log spectrogram)")
axs[0].set_xlabel("time frames")
axs[0].set_ylabel("frequency bins")

axs[1].imshow(nv, aspect="auto", origin="lower", cmap="gray", vmin=-1, vmax=1)
axs[1].set_title("Vertical Edge kernel (highlights time-direction changes)")
axs[1].set_xlabel("time frames")
axs[1].set_ylabel("frequency bins")

axs[2].imshow(nh, aspect="auto", origin="lower", cmap="gray", vmin=-1, vmax=1)
axs[2].set_title("Horizontal Edge kernel (highlights freq-direction changes)")
axs[2].set_xlabel("time frames")
axs[2].set_ylabel("frequency bins")

plt.suptitle("2D Convolution on Audio Spectrogram (Kernel as Detail Extractor)", fontsize=18)
plt.show()

phase = np.angle(Z)    # Z 是你 stft 得到的复数谱
# 用卷积结果构造“新幅度”
mag_v = np.abs(out_v)

# 归一化，防止爆音
mag_v = mag_v / (np.max(mag_v) + 1e-8)

# 构造新的复数谱
Z_v = mag_v * np.exp(1j * phase)
_, x_v = istft(
    Z_v,
    fs=sr,
    nperseg=win,
    noverlap=noverlap,
    nfft=n_fft
)

x_v = x_v / (np.max(np.abs(x_v)) + 1e-9)

sf.write("audio_vertical_edge.wav", x_v, sr)
