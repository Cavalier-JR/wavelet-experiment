"""复现结果核对。仅依赖Python标准库，不重新计算实验。"""
from pathlib import Path
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent

def main():
    expected = json.loads((ROOT/'reference/results_expected.json').read_text(encoding='utf-8'))
    actual = json.loads((ROOT/'data/results.json').read_text(encoding='utf-8'))
    count = 0
    def compare(a, b, location):
        nonlocal count
        if isinstance(a, dict):
            assert set(a) == set(b), f'字段不一致: {location}'
            for key in a: compare(a[key], b[key], f'{location}.{key}')
        elif isinstance(a, list):
            assert len(a) == len(b), f'记录数不一致: {location}'
            for i, (x, y) in enumerate(zip(a, b)): compare(x, y, f'{location}[{i}]')
        elif isinstance(a, (int, float)):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), f'数值不一致: {location}: {a} vs {b}'
            count += 1
        else:
            assert a == b, f'内容不一致: {location}'
    for group in ['denoising','sensitivity','detection']:
        compare(expected[group],actual[group],group)
    for name in ['base_seed','sampling_rate_hz','length','repeats','true_events_samples','true_events_seconds']:
        compare(expected['environment'][name],actual['environment'][name],name)
    # 从逐次CSV独立计算均值和样本标准差，避免只核对汇总文件。
    for kind,keys,nrows in [('denoising',['signal','input_snr_nominal','method'],1080),
                           ('sensitivity',['wavelet','level'],240)]:
        with (ROOT/f'data/{kind}_raw.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==nrows, f'{kind}逐次记录数异常'
        for record in actual[kind]:
            selected=[r for r in rows if all(str(r[k])==str(record[k]) for k in keys)]
            assert len(selected)==30
            for metric in ['snr','rmse','corr']:
                values=[float(r[metric]) for r in selected]
                for suffix,value in [('mean',statistics.mean(values)),('std',statistics.stdev(values))]:
                    assert math.isclose(value,record[metric+'_'+suffix],rel_tol=1e-9,abs_tol=1e-9)
    for name in ['fig1_signals.png','fig2_denoising.png','fig3_snr.png','fig4_sensitivity.png','fig5_detection.png']:
        p=ROOT/'figures'/name
        assert p.is_file() and p.stat().st_size>1000, f'缺少图像: {name}'
    # 检测统计也从逐次预测位置重新核对，避免仅对比两个汇总文件。
    with (ROOT/'data/detection_raw.csv').open(encoding='utf-8-sig',newline='') as f:
        detections=list(csv.DictReader(f))
    assert len(detections)==180, '检测逐次记录数异常'
    events=actual['environment']['true_events_samples']
    fs=actual['environment']['sampling_rate_hz']
    for record in actual['detection']:
        selected=[r for r in detections if int(r['input_snr_nominal'])==record['input_snr_nominal'] and r['method']==record['method']]
        assert len(selected)==30
        all_errors=[]
        for row in selected:
            peaks=[int(v) for v in row['peaks_samples'].split()]
            used=set()
            errors=[]
            for event in events:
                available=[(abs(p-event),i) for i,p in enumerate(peaks) if i not in used]
                if available:
                    error,index=min(available)
                    if error<=16:
                        used.add(index)
                        errors.append(error/fs*1000)
            assert (len(errors),len(peaks)-len(errors),len(events)-len(errors)) == tuple(int(row[k]) for k in ['tp','fp','fn'])
            all_errors.extend(errors)
        for key in ['tp','fp','fn','negative_fp']:
            assert sum(int(r[key]) for r in selected)==record[key], key
        assert sum(int(r['negative_fp'])>0 for r in selected)==record['negative_alarm_windows']
        tp,fp,fn=(record[k] for k in ['tp','fp','fn'])
        for key,value in [('precision',tp/max(tp+fp,1)),('recall',tp/max(tp+fn,1)),
                          ('f1',2*tp/max(2*tp+fp+fn,1))]:
            assert math.isclose(value,record[key],rel_tol=1e-9,abs_tol=1e-9),key
        if all_errors:
            assert math.isclose(statistics.mean(all_errors),record['mae_ms'],rel_tol=1e-9,abs_tol=1e-9)
    print(f'PASS：{count}个数值与原报告结果一致（容差1e-9）。')
    print('PASS：1080条消噪逐次记录、240条参数逐次记录的均值和标准差一致。')
    print('PASS：180条检测逐次记录的事件匹配、F1、定位误差及背景告警统计一致。')
    print('PASS：5张实验图均已生成。图像外观受系统字体影响，不要求文件字节相同。')
    print('核对完成。请另行打开图片，确认中文、图例和曲线显示正常。')

if __name__=='__main__':
    try: main()
    except (AssertionError,FileNotFoundError,KeyError,ValueError) as exc:
        print(f'FAIL：{exc}')
        raise SystemExit(1)
