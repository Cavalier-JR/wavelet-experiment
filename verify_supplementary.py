"""独立从CSV复算补充实验统计，并核对选参、种子与主实验的对应关系。"""
from pathlib import Path
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent


def read_csv(name):
    with (ROOT / 'data' / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def main():
    result = json.loads((ROOT / 'data/supplementary_results.json').read_text(encoding='utf-8'))
    validation = read_csv('baseline_validation_raw.csv')
    test = read_csv('supplementary_raw.csv')
    primary = read_csv('denoising_raw.csv')
    assert len(validation) == 300 and len(test) == 150
    validation_seeds = {int(r['seed']) for r in validation}
    test_seeds = {int(r['seed']) for r in test}
    assert len(validation_seeds) == len(test_seeds) == 30
    assert validation_seeds.isdisjoint(test_seeds)
    for raw, summaries, keys, metrics in [
            (validation, result['validation'], ['family', 'parameter'], ['snr', 'rmse', 'corr']),
            (test, result['test'], ['method'], ['snr', 'rmse', 'corr', 'local_rmse', 'transient_gain'])]:
        for summary in summaries:
            selected = [r for r in raw if all(str(r[k]) == str(summary[k]) for k in keys)]
            assert len(selected) == summary['n'] == 30
            assert {int(r['repeat']) for r in selected} == set(range(30))
            for metric in metrics:
                values = [float(r[metric]) for r in selected]
                close(statistics.mean(values), summary[metric + '_mean'])
                close(statistics.stdev(values), summary[metric + '_std'])
    for family, parameter in result['selected_parameters'].items():
        best = max((s for s in result['validation'] if s['family'] == family), key=lambda s: s['snr_mean'])
        assert parameter == best['parameter']
    for method, main_method in [('moving_average_17', 'moving_average'), ('db4_hard', 'db4_hard'), ('db4_soft', 'db4_soft')]:
        records = [r for r in test if r['method'] == method]
        for row in records:
            original = next(r for r in primary if r['signal'] == 'S3' and r['input_snr_nominal'] == '10' and r['method'] == main_method and r['repeat'] == row['repeat'])
            assert original['seed'] == row['seed']
            for metric in ['snr', 'rmse', 'corr']:
                close(float(row[metric]), float(original[metric]))
    assert result['protocol']['local_samples'] == 123
    image = ROOT / 'figures/fig6_supplementary.png'
    assert image.is_file() and image.stat().st_size > 1000
    print('PASS：300条验证指标、150条测试指标的均值和样本标准差已独立复算。')
    print('PASS：验证集与测试集种子不重叠；参数只由验证集确定；测试数据与主实验一致。')
    print('PASS：补充图6已生成。')


if __name__ == '__main__':
    main()
