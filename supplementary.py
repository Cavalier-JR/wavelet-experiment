"""补充公平对照和局部误差评价。运行：python supplementary.py。

主实验参数和原有数据不变。新增独立噪声验证集选择移动平均窗口与
FFT 截止频率，然后仅在原有 S3 10 dB 测试记录上报告比较结果。
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import t as student_t
import pywt
import matplotlib.pyplot as plt
import experiment as ex

ROOT = Path(__file__).resolve().parent
WINDOWS = (5, 9, 17, 33, 65)
CUTOFFS = (16, 32, 64, 96, 128)
VALIDATION_SEED = ex.BASE_SEED + 61000
TEST_SEED = ex.BASE_SEED + 21000


def fft_lowpass(y, cutoff):
    # 两端各对称延拓 N 点，避免直接把原序列首尾连接；保留 DC 和负频率对称性。
    padded = np.pad(y, (len(y), len(y)), mode='symmetric')
    spectrum = np.fft.rfft(padded)
    frequency = np.fft.rfftfreq(len(padded), 1 / ex.FS)
    spectrum[frequency > cutoff] = 0
    return np.fft.irfft(spectrum, n=len(padded))[len(y):2 * len(y)]


def paired_ci(a, b):
    delta = np.asarray(a) - np.asarray(b)
    mean = float(delta.mean())
    half = float(student_t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
    return {'n': len(delta), 'difference_mean': mean, 'ci95_low': mean - half,
            'ci95_high': mean + half, 'difference_std': float(delta.std(ddof=1))}


def run():
    signals = ex.clean_signals()
    x = signals['S3']
    local = (ex.T >= 1) & (ex.T < 1.12)
    burst = x[local] - signals['S2'][local]
    validation = []
    for repeat in range(ex.REPEATS):
        y = ex.add_noise(x, 10, VALIDATION_SEED + repeat)
        for family, values in [('moving_average', WINDOWS), ('fft_lowpass', CUTOFFS)]:
            for value in values:
                z = uniform_filter1d(y, size=value, mode='reflect') if family == 'moving_average' else fft_lowpass(y, value)
                validation.append({'family': family, 'parameter': value, 'repeat': repeat,
                                   'seed': VALIDATION_SEED + repeat, **ex.metrics(x, z)})
    validation_summary = ex.summarize(validation, ['family', 'parameter'], ['snr', 'rmse', 'corr'])
    chosen = {family: max((r for r in validation_summary if r['family'] == family), key=lambda r: r['snr_mean'])['parameter']
              for family in ['moving_average', 'fft_lowpass']}
    method_order = ['moving_average_17', 'moving_average_validated', 'fft_validated', 'db4_hard', 'db4_soft']
    rows = []
    for repeat in range(ex.REPEATS):
        y = ex.add_noise(x, 10, TEST_SEED + repeat)
        outputs = {'moving_average_17': uniform_filter1d(y, size=17, mode='reflect'),
                   'moving_average_validated': uniform_filter1d(y, size=chosen['moving_average'], mode='reflect'),
                   'fft_validated': fft_lowpass(y, chosen['fft_lowpass']),
                   'db4_hard': ex.denoise(y, kind='hard'),
                   'db4_soft': ex.denoise(y, kind='soft')}
        for method in method_order:
            z = outputs[method]
            rows.append({'method': method, 'repeat': repeat, 'seed': TEST_SEED + repeat,
                         **ex.metrics(x, z),
                         'local_rmse': float(np.sqrt(np.mean((x[local] - z[local]) ** 2))),
                         # 以无噪 S2 为背景扣除参考，仅作评价，不用于重建或参数选择。
                         'transient_gain': float(np.dot(z[local] - signals['S2'][local], burst) / np.dot(burst, burst))})
    summary = ex.summarize(rows, ['method'], ['snr', 'rmse', 'corr', 'local_rmse', 'transient_gain'])
    def values(method, metric='snr'):
        return [r[metric] for r in rows if r['method'] == method]
    paired = {f'hard_minus_{method}': paired_ci(values('db4_hard'), values(method))
              for method in ['moving_average_17', 'moving_average_validated', 'fft_validated']}
    with (ROOT / 'data/sensitivity_raw.csv').open(encoding='utf-8-sig', newline='') as f:
        sensitivity = list(csv.DictReader(f))
    def sens(wavelet, level):
        records = sorted((r for r in sensitivity if r['wavelet'] == wavelet and int(r['level']) == level), key=lambda r: int(r['repeat']))
        assert len(records) == ex.REPEATS
        return [float(r['snr']) for r in records]
    paired['db2_L5_minus_db4_L5'] = paired_ci(sens('db2', 5), sens('db4', 5))
    paired['db4_L6_minus_db4_L5'] = paired_ci(sens('db4', 6), sens('db4', 5))
    gain = abs(np.sin(np.pi * 17 * 70 / ex.FS) / (17 * np.sin(np.pi * 70 / ex.FS)))
    reconstruction = {}
    for name, signal in signals.items():
        c = pywt.wavedec(signal, 'db4', mode='symmetric', level=5)
        reconstructed = pywt.waverec(c, 'db4', mode='symmetric')[:ex.N]
        reconstruction[name] = float(np.max(np.abs(signal - reconstructed)))
    ex.write_csv('baseline_validation_raw.csv', validation)
    ex.write_csv('baseline_validation_summary.csv', validation_summary)
    ex.write_csv('supplementary_raw.csv', rows)
    ex.write_csv('supplementary_summary.csv', summary)
    result = {'protocol': {'signal': 'S3', 'nominal_snr_db': 10,
                          'validation_repeats': ex.REPEATS, 'validation_seed_start': VALIDATION_SEED,
                          'test_repeats': ex.REPEATS, 'test_seed_start': TEST_SEED,
                          'moving_average_candidates': WINDOWS, 'fft_cutoff_candidates_hz': CUTOFFS,
                          'selection_metric': 'mean validation output SNR',
                          'fft_padding_points_each_side': ex.N,
                          'local_interval_seconds': [1, 1.12], 'local_samples': int(local.sum())},
              'selected_parameters': chosen, 'validation': validation_summary, 'test': summary,
              'paired_snr_intervals': paired,
              'ma17_gain_at_70hz': float(gain), 'ma17_gain_db_at_70hz': float(20 * np.log10(gain)),
              'negative_alarm_probability_one_sided_95_upper_n30': float(1 - .05 ** (1 / 30)),
              'unmodified_db4_max_abs_reconstruction_error': reconstruction,
              'environment': {'numpy': np.__version__, 'PyWavelets': pywt.__version__}}
    (ROOT / 'data/supplementary_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for family, color, marker in [('moving_average', '#235D91', 'o'), ('fft_lowpass', '#44866A', 's')]:
        ss = [r for r in validation_summary if r['family'] == family]
        axes[0].plot(range(len(ss)), [r['snr_mean'] for r in ss], marker=marker, color=color,
                     label='移动平均窗口' if family == 'moving_average' else 'FFT截止频率')
    axes[0].set_xticks(range(5), ['5 / 16', '9 / 32', '17 / 64', '33 / 96', '65 / 128'])
    axes[0].set_xlabel('候选值  窗长 / 截止频率Hz', fontsize=9)
    axes[0].set_ylabel('验证集输出SNR / dB')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=.18)
    axes[1].bar(range(len(summary)), [r['local_rmse_mean'] for r in summary],
                yerr=[r['local_rmse_std'] for r in summary], capsize=3,
                color=['#B9BDC2', '#C27937', '#7D9FAD', '#235D91', '#44866A'])
    axes[1].set_xticks(range(len(summary)), ['MA17', f'MA{chosen["moving_average"]}', f'FFT{chosen["fft_lowpass"]}', '硬阈值', '软阈值'], fontsize=8)
    axes[1].set_ylabel('暂态区间RMSE')
    axes[1].set_xlabel('独立验证选参后在原测试集评价', fontsize=9)
    axes[1].grid(axis='y', alpha=.18)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/fig6_supplementary.png')
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
