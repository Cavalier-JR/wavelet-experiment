"""小波课程实验：仿真信号、消噪、参数分析与阶跃定位。
运行：python experiment.py。所有输出相对本脚本保存，随机种子固定。
"""
from pathlib import Path
import csv
import json
import platform
import numpy as np
import scipy
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
DATA = ROOT / 'data'
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
FS, N, REPEATS, BASE_SEED = 1024, 2048, 30, 20261104
T = np.arange(N) / FS
EVENTS = np.array([666, 1382])
METHODS = ['raw', 'moving_average', 'db4_hard', 'db4_soft']
COLORS = {'raw':'#AEB5BC', 'moving_average':'#C27937', 'db4_hard':'#235D91', 'db4_soft':'#44866A'}
LABELS = {'raw':'含噪信号','moving_average':'17点移动平均','db4_hard':'db4硬阈值','db4_soft':'db4软阈值'}
font = Path('/usr/local/share/fonts/course-papers/NotoSerifCJKsc-Regular.otf')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
else:
    available = {item.name for item in font_manager.fontManager.ttflist}
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Serif CJK SC',
                  'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK JP',
                  'WenQuanYi Micro Hei', 'Arial Unicode MS']
    selected = next((name for name in candidates if name in available), None)
    if selected:
        plt.rcParams['font.family'] = selected
    else:
        import warnings
        warnings.warn('未找到中文字体。请安装微软雅黑或Noto CJK字体后重新运行；数值计算不受影响。')
        plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams.update({'font.size':10, 'axes.unicode_minus':False, 'axes.spines.top':False,
                     'axes.spines.right':False, 'savefig.dpi':220, 'figure.dpi':130})

def clean_signals():
    s1 = 0.2*np.sin(2*np.pi*1.5*T) + 0.08*np.cos(2*np.pi*4*T)
    step = 0.7*(np.arange(N)>=EVENTS[0])-0.5*(np.arange(N)>=EVENTS[1])
    s2 = s1+step
    tau = T-1.0
    burst = np.where((tau>=0)&(tau<0.12), 0.6*np.exp(-35*np.maximum(tau,0))*np.sin(2*np.pi*70*tau),0)
    return {'S1':s1,'S2':s2,'S3':s2+burst}

def add_noise(x, snr, seed, reference_power=None):
    power = np.mean(x*x) if reference_power is None else reference_power
    sigma = np.sqrt(power / 10**(snr/10))
    noise = np.random.default_rng(seed).normal(0,sigma,N)
    return x+noise

def estimate_sigma(y):
    c = pywt.wavedec(y,'db4',mode='symmetric',level=1)[-1]
    return float(np.median(np.abs(c-np.median(c)))/0.6744897501960817)

def denoise(y, wavelet='db4', level=5, kind='hard'):
    c = pywt.wavedec(y,wavelet,mode='symmetric',level=level)
    sigma = np.median(np.abs(c[-1]-np.median(c[-1])))/0.6744897501960817
    threshold = sigma*np.sqrt(2*np.log(len(y)))
    out = [c[0]]+[pywt.threshold(d,threshold,mode=kind) for d in c[1:]]
    return pywt.waverec(out,wavelet,mode='symmetric')[:len(y)]

def process(y):
    return {'raw':y,'moving_average':uniform_filter1d(y,size=17,mode='reflect'),
            'db4_hard':denoise(y,kind='hard'),'db4_soft':denoise(y,kind='soft')}

def metrics(x,z):
    mse=float(np.mean((x-z)**2))
    return {'snr':float(10*np.log10(np.sum(x*x)/np.sum((x-z)**2))),
            'rmse':float(np.sqrt(mse)), 'corr':float(np.corrcoef(x,z)[0,1])}

def write_csv(name, rows):
    with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows, keys, vals):
    groups={}
    for r in rows:
        key=tuple(r[k] for k in keys);groups.setdefault(key,[]).append(r)
    out=[]
    for key,group in groups.items():
        d=dict(zip(keys,key));d['n']=len(group)
        for v in vals:
            a=np.array([g[v] for g in group]);d[v+'_mean']=float(a.mean());d[v+'_std']=float(a.std(ddof=1))
        out.append(d)
    return out

def haar_score(z,sigma):
    """全时间平移的Haar左右窗对比，m为每侧点数，不作二抽取。"""
    cs=np.r_[0.,np.cumsum(z)]
    score=np.zeros(len(z))
    for m in [8,16,32]:
        idx=np.arange(m,len(z)-m+1)
        right=cs[idx+m]-cs[idx]
        left=cs[idx]-cs[idx-m]
        coeff=(right-left)/np.sqrt(2*m)
        # 同时考虑原始噪声和本尺度背景变化，避免低噪声时平滑趋势被放大。
        scale_mad=np.median(np.abs(coeff-np.median(coeff)))/0.6744897501960817
        score[idx]=np.maximum(score[idx],np.abs(coeff)/max(sigma,scale_mad,1e-12))
    score[:64]=0;score[-64:]=0
    peaks,_=find_peaks(score,height=6,distance=128)
    return score,peaks

def match(peaks,events,tolerance=16):
    used=set();errors=[]
    # 候选按时间顺序，每个真实事件与预测最多各匹配一次。
    for event in events:
        available=[(abs(int(p)-int(event)),i) for i,p in enumerate(peaks) if i not in used]
        if available:
            err,idx=min(available)
            if err<=tolerance:used.add(idx);errors.append(err/FS*1000)
    return len(errors),len(peaks)-len(errors),len(events)-len(errors),errors

def run():
    signals=clean_signals();rows=[]
    for si,(name,x) in enumerate(signals.items()):
        for snr in [5,10,15]:
            for r in range(REPEATS):
                seed=BASE_SEED+si*10000+snr*100+r
                y=add_noise(x,snr,seed)
                for method,z in process(y).items():
                    rows.append({'signal':name,'input_snr_nominal':snr,'repeat':r,'seed':seed,
                                 'method':method,**metrics(x,z)})
    write_csv('denoising_raw.csv',rows)
    summary=summarize(rows,['signal','input_snr_nominal','method'],['snr','rmse','corr'])
    write_csv('denoising_summary.csv',summary)
    sensitivity=[]
    for r in range(REPEATS):
        y=add_noise(signals['S3'],10,BASE_SEED+20000+1000+r)
        for wave in ['haar','db2','db4','sym4']:
            for level in ([3,4,5,6,7] if wave=='db4' else [5]):
                z=denoise(y,wavelet=wave,level=level)
                sensitivity.append({'wavelet':wave,'level':level,'repeat':r,**metrics(signals['S3'],z)})
    write_csv('sensitivity_raw.csv',sensitivity)
    sens=summarize(sensitivity,['wavelet','level'],['snr','rmse','corr'])
    write_csv('sensitivity_summary.csv',sens)
    detection=[];det_summary=[]
    for snr in [5,10,15]:
        stats={m:{'tp':0,'fp':0,'fn':0,'errors':[],'negative_fp':0,'negative_alarm':0} for m in ['raw','db4_hard']}
        for r in range(REPEATS):
            y=add_noise(signals['S2'],snr,BASE_SEED+40000+snr*100+r)
            yn=add_noise(signals['S1'],snr,BASE_SEED+50000+snr*100+r,reference_power=np.mean(signals['S2']**2))
            for method in stats:
                z=y if method=='raw' else denoise(y)
                zn=yn if method=='raw' else denoise(yn)
                score,p=haar_score(z,estimate_sigma(y))
                scoren,pn=haar_score(zn,estimate_sigma(yn))
                tp,fp,fn,errors=match(p,EVENTS)
                s=stats[method];s['tp']+=tp;s['fp']+=fp;s['fn']+=fn;s['errors']+=errors
                s['negative_fp']+=len(pn);s['negative_alarm']+=int(len(pn)>0)
                detection.append({'input_snr_nominal':snr,'repeat':r,'method':method,'tp':tp,'fp':fp,'fn':fn,
                                  'peaks_samples':' '.join(map(str,p)), 'negative_fp':len(pn)})
        for method,s in stats.items():
            precision=s['tp']/max(s['tp']+s['fp'],1);recall=s['tp']/max(s['tp']+s['fn'],1)
            det_summary.append({'input_snr_nominal':snr,'method':method,'tp':s['tp'],'fp':s['fp'],'fn':s['fn'],
                                'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-12),
                                'mae_ms':float(np.mean(s['errors'])) if s['errors'] else None,
                                'negative_fp':s['negative_fp'],'negative_alarm_windows':s['negative_alarm'],'negative_windows':REPEATS})
    write_csv('detection_raw.csv',detection);write_csv('detection_summary.csv',det_summary)

    # 图1：三类原始信号，图2：S3消噪效果及70 Hz暂态局部细节。
    fig,ax=plt.subplots(3,1,figsize=(7.2,4.4),sharex=True)
    for a,(name,x) in zip(ax,signals.items()):
        a.plot(T,x,color='#235D91',lw=1);a.set_ylabel(name+' 幅值');a.grid(alpha=.18)
    ax[-1].set_xlabel('时间 / s');fig.tight_layout();fig.savefig(FIG/'fig1_signals.png');plt.close(fig)
    y=add_noise(signals['S3'],10,BASE_SEED+21000);zs=process(y)
    np.savez(DATA/'example_signals.npz',time=T,**signals,noisy_S3=y,**{k:v for k,v in zs.items() if k!='raw'})
    fig,ax=plt.subplots(2,1,figsize=(7.2,4.8))
    for a in ax:
        a.plot(T,y,c=COLORS['raw'],lw=.5,alpha=.6,label=LABELS['raw'])
        a.plot(T,signals['S3'],c='#222222',lw=1.2,label='无噪参考')
        for m in METHODS[1:]:a.plot(T,zs[m],color=COLORS[m],lw=1,label=LABELS[m])
        a.set_ylabel('归一化幅值');a.grid(alpha=.18)
    ax[0].legend(ncol=3,fontsize=8,loc='upper right');ax[0].set_xlim(0,2)
    ax[1].set_xlim(.97,1.10);ax[1].set_xlabel('时间 / s');fig.tight_layout();fig.savefig(FIG/'fig2_denoising.png');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.8),sharey=True)
    for a,name in zip(ax,signals):
        for m in METHODS[1:]:
            rr=[v for v in summary if v['signal']==name and v['method']==m]
            a.errorbar([v['input_snr_nominal'] for v in rr],[v['snr_mean'] for v in rr],
                       yerr=[v['snr_std'] for v in rr],marker='o',ms=3,c=COLORS[m],capsize=2,label=LABELS[m])
        a.set_title(name);a.set_xticks([5,10,15]);a.set_xlabel('输入信噪比 / dB');a.grid(alpha=.18)
    ax[0].set_ylabel('输出信噪比 / dB');ax[1].legend(fontsize=7,loc='upper left');fig.tight_layout();fig.savefig(FIG/'fig3_snr.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    ss=sorted([v for v in sens if v['wavelet']=='db4'],key=lambda x:x['level'])
    ax[0].errorbar([v['level'] for v in ss],[v['snr_mean'] for v in ss],yerr=[v['snr_std'] for v in ss],marker='o',capsize=3,c='#235D91');ax[0].set_xlabel('db4 分解层数');ax[0].set_ylabel('输出信噪比 / dB');ax[0].set_xticks([3,4,5,6,7])
    ss=[next(v for v in sens if v['wavelet']==w and v['level']==5) for w in ['haar','db2','db4','sym4']]
    ax[1].bar([v['wavelet'] for v in ss],[v['snr_mean'] for v in ss],yerr=[v['snr_std'] for v in ss],color=['#9EAAA6','#7D9FAD','#235D91','#44866A'],capsize=3);ax[1].set_xlabel('小波基  分解层数固定为5');ax[1].set_ylabel('输出信噪比 / dB')
    for a in ax:a.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(FIG/'fig4_sensitivity.png');plt.close(fig)
    yy=add_noise(signals['S2'],10,BASE_SEED+41000);zz=denoise(yy)
    score,p=haar_score(yy,estimate_sigma(yy))
    score_filtered,_=haar_score(zz,estimate_sigma(yy))
    fig,ax=plt.subplots(2,1,figsize=(7.2,4),sharex=True)
    ax[0].plot(T,yy,c=COLORS['raw'],lw=.5,alpha=.6,label='含噪信号');ax[0].plot(T,zz,c='#235D91',lw=1,label='db4硬阈值');ax[0].set_ylabel('归一化幅值');ax[0].legend(ncol=2,fontsize=8)
    ax[1].plot(T,score,c='#235D91',lw=1,label='原始信号得分');ax[1].plot(T,score_filtered,c='#44866A',lw=1,label='消噪信号得分');ax[1].axhline(6,c='#C27937',ls='--',lw=1,label='阈值6');ax[1].plot(T[p],score[p],'o',ms=4,c='#C27937',label='检测点');ax[1].set_ylabel('标准化得分');ax[1].set_xlabel('时间 / s');ax[1].legend(ncol=4,fontsize=7)
    for a in ax:
        for e in EVENTS:a.axvline(T[e],c='#555555',ls=':',lw=.8)
        a.grid(alpha=.18)
    fig.tight_layout();fig.savefig(FIG/'fig5_detection.png');plt.close(fig)
    env={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
         'PyWavelets':pywt.__version__,'matplotlib':matplotlib.__version__,'base_seed':BASE_SEED,
         'sampling_rate_hz':FS,'length':N,'repeats':REPEATS,'true_events_samples':EVENTS.tolist(),
         'true_events_seconds':T[EVENTS].tolist()}
    result={'environment':env,'denoising':summary,'sensitivity':sens,'detection':det_summary}
    (DATA/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'environment':env,'S3_10dB':[s for s in summary if s['signal']=='S3' and s['input_snr_nominal']==10],
                      'sensitivity':sens,'detection':det_summary},ensure_ascii=False,indent=2))

if __name__=='__main__':
    run()
