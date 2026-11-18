"""将实验JSON输出为与正文相对应的表格。"""
from pathlib import Path
import json
r=json.loads((Path(__file__).resolve().parent/'data/results.json').read_text(encoding='utf-8'))
print('表2：10 dB条件  signal method SNR均值 SNR标准差 RMSE 相关系数')
for x in r['denoising']:
    if x['input_snr_nominal']==10:
        print(f"{x['signal']:3} {x['method']:16} {x['snr_mean']:7.2f} {x['snr_std']:6.2f} {x['rmse_mean']:.4f} {x['corr_mean']:.4f}")
print('\n图4：小波基 层数 输出SNR均值')
for x in r['sensitivity']: print(f"{x['wavelet']:5} {x['level']} {x['snr_mean']:.2f}")
print('\n表3：输入dB 方法 检出数/60 召回率 F1 定位误差ms 背景告警/30')
for x in r['detection']:
    print(f"{x['input_snr_nominal']:2} {x['method']:10} {x['tp']:2}/60 {100*x['recall']:6.1f}% {x['f1']:.3f} {x['mae_ms']:.3f} {x['negative_alarm_windows']}/30")
