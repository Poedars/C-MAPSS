"""教材新增的流程衔接代码；核心算法直接调用core.py中的开源函数。"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
from pathlib import Path
import argparse, json, random, hashlib, platform
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from core import (calc_training_rul, split_data,
    process_input_data_with_targets, create_compiled_model,
    scheduler, compute_s_score)

FEATURES = [f's{i}' for i in
    [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
CAP, WINDOW, SEED = 125, 30, 42

def read_frame(path):
    df = pd.read_csv(path, sep=r'\s+', header=None)
    if df.shape[1] != 26:
        raise ValueError('需要26列数据')
    df.columns = ['id','cycle','op1','op2','op3'] + [f's{i}' for i in range(1,22)]
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError('存在非有限数值')
    df = df.sort_values(['id','cycle']).reset_index(drop=True)
    for _, g in df.groupby('id'):
        if not np.all(np.diff(g.cycle) == 1):
            raise ValueError('周期不连续或重复')
    return df

def prepare(data_dir):
    train = calc_training_rul(read_frame(data_dir/'train_FD001.txt'))
    train['RUL'] = train['RUL'].clip(upper=CAP)
    test = read_frame(data_dir/'test_FD001.txt')
    truth = np.loadtxt(data_dir/'RUL_FD001.txt', ndmin=1)
    random.seed(SEED)  # split_data使用Python random.shuffle
    fit, val = split_data(train, randomize=True, train_pct=0.8)
    fit, val = fit.copy(), val.copy()
    scaler = StandardScaler().fit(fit[FEATURES])
    for frame in (fit,val,test):
        frame[FEATURES] = scaler.transform(frame[FEATURES])
    def windows(frame):
        xs,ys = [],[]
        for _,g in frame.groupby('id', sort=True):
            if len(g)<WINDOW: raise ValueError('序列不足30周期')
            x,y = process_input_data_with_targets(
                g[FEATURES].to_numpy(dtype=np.float32),
                g.RUL.to_numpy(dtype=np.float32),WINDOW,1)
            xs.append(x);ys.append(y)
        return np.concatenate(xs).astype('float32'),np.concatenate(ys).astype('float32')
    x_train,y_train = windows(fit)
    x_val,y_val = windows(val)
    xs,ids = [],[]
    for unit,g in test.groupby('id', sort=True):
        if len(g)<WINDOW: raise ValueError('测试序列不足30周期')
        xs.append(g[FEATURES].to_numpy(dtype=np.float32)[-WINDOW:])
        ids.append(int(unit))
    if ids != list(range(1,len(truth)+1)):
        raise ValueError('测试ID与RUL行号不对应')
    x_test=np.stack(xs)
    return (train,fit,val,scaler,x_train,y_train,x_val,y_val,x_test,truth,ids)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('CMAPSSData'))
    ap.add_argument('--out',type=Path,default=Path('output'))
    ap.add_argument('--epochs',type=int,default=10)
    args=ap.parse_args()
    if args.epochs<1: raise ValueError('epochs必须为正整数')
    args.out.mkdir(parents=True,exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    (raw,fit,val,scaler,xt,yt,xv,yv,xe,truth,ids)=prepare(args.data_dir)
    # 适配受限CPU环境：只改变输入容器，不改变开源模型和优化器。
    def dataset(x,y,shuffle=False):
        ds=tf.data.Dataset.from_tensor_slices((x,y))
        if shuffle: ds=ds.shuffle(len(x),seed=SEED)
        opts=tf.data.Options();opts.threading.private_threadpool_size=1
        return ds.batch(128).with_options(opts)
    model=create_compiled_model()
    callback=tf.keras.callbacks.LearningRateScheduler(scheduler)
    history=model.fit(dataset(xt,yt,True),epochs=args.epochs,
        validation_data=dataset(xv,yv),callbacks=[callback],verbose=2)
    # 与上游最后窗口评价一致：不裁剪预测，不截断测试真实标签。
    pred=model(xe,training=False).numpy().reshape(-1)
    rmse=np.sqrt(mean_squared_error(truth,pred))
    score=compute_s_score(truth,pred)
    report={'RMSE':float(rmse),'Score':float(score),
        'epochs':args.epochs,'parameters':model.count_params(),
        'train_shape':list(xt.shape),'val_shape':list(xv.shape),'test_shape':list(xe.shape),
        'train_ids':sorted(map(int,fit.id.unique())),'val_ids':sorted(map(int,val.id.unique())),
        'window':WINDOW,'cap':CAP,'seed':SEED,
        'versions':{'python':platform.python_version(),'tensorflow':tf.__version__,
            'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.data_dir.glob('*FD001.txt')}}
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(history.history).to_csv(args.out/'history.csv',index=False)
    pd.DataFrame({'unit':ids,'true_rul':truth,'pred_rul':pred,'error':pred-truth}).to_csv(args.out/'predictions.csv',index=False)
    model.save(args.out/'model.keras')
    np.savez(args.out/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_,features=FEATURES)
    # 绘图单独实现，使用同一批数据与实际结果。
    from plot_results import draw_source_figures
    draw_source_figures(raw,history.history,truth,pred,args.out)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
