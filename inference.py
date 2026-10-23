"""Đọc cấu hình web cũ/V4; suy luận thật, không huấn luyện hoặc gán điểm."""
import copy
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
PREPROCESS = {'raw_0_255':'RGB 0–255; chuẩn hóa trong model',
              'scale_0_1':'Chia 255 ngoài model',
              'resnet50':'ResNet50 preprocess_input ngoài model',
              'vgg16':'VGG16 preprocess_input ngoài model',
              'vgg19':'VGG19 preprocess_input ngoài model'}
RESIZE = {'tensorflow_bilinear':'tensorflow_bilinear'}
NAMES = ['CustomCNN','ResNet50','VGG16','VGG19']

def default_spec(name):
    return dict(id=name,name='Custom CNN' if name=='CustomCNN' else name,
                path=f'models/{name}_best.keras',class_names=['Normal','Stroke'],
                preprocessing='raw_0_255',interpolation='tensorflow_bilinear',
                apply_exif_orientation=False,binary_output_index=1,
                output_type='probabilities',decision_threshold=0.5,dynamic_size=[224,224])

def load_config(path=None):
    path=Path(path) if path else BASE/'model_config.json'
    if path.is_file():
        cfg=json.loads(path.read_text(encoding='utf-8-sig'))
    else:
        cfg={'configuration_source':'Mặc định cho model V4; chưa thấy model_config.json',
             'models':[default_spec(n) for n in NAMES]}
    raw=cfg['models']
    if isinstance(raw,dict):
        entries=[dict(value,id=key,name=value.get('name',key)) for key,value in raw.items()]
    elif isinstance(raw,list):
        entries=copy.deepcopy(raw)
    else:
        raise ValueError('models cần là danh sách web cũ hoặc dictionary từ notebook V4.')
    result=[]
    for entry in entries:
        mid=entry.get('id',entry.get('name'))
        if not isinstance(mid,str) or not mid:
            raise ValueError('Thiếu id model.')
        spec=default_spec(mid)
        spec.update(entry)
        if 'threshold' in entry:
            spec['decision_threshold']=entry['threshold']
        if 'output_mode' in entry:
            spec['output_type']=entry['output_mode']
        if 'dynamic_input_size' in entry:
            spec['dynamic_size']=entry['dynamic_input_size']
        if spec['preprocessing'] not in PREPROCESS:
            raise ValueError(f"Chuẩn hóa chưa hỗ trợ: {spec['preprocessing']}. Dùng config xuất từ V4.")
        if spec['interpolation'] not in RESIZE:
            raise ValueError('Cần tensorflow_bilinear để khớp notebook V4.')
        if len(spec['class_names'])!=2 or len(set(spec['class_names']))!=2:
            raise ValueError('Cấu hình phải có hai nhãn khác nhau.')
        if spec['binary_output_index'] not in [0,1]:
            raise ValueError('binary_output_index phải là 0 hoặc 1.')
        if not 0<float(spec['decision_threshold'])<1:
            raise ValueError('Ngưỡng phải nằm trong (0,1).')
        if spec['output_type'] not in ['probabilities','logits']:
            raise ValueError('output_type phải là probabilities hoặc logits.')
        if spec.get('apply_exif_orientation',False):
            raise ValueError('V4 không xoay EXIF; hãy kiểm tra lại cấu hình training.')
        result.append(spec)
    if not any(str(s['id']).replace('_','').lower()=='vgg19' for s in result):
        result.append(default_spec('VGG19'))
    if len({s['id'] for s in result})!=len(result):
        raise ValueError('Trùng id model.')
    return dict(configuration_source=cfg.get('configuration_source','model_config.json'),models=result)

def resolve_model(spec):
    path=Path(spec['path']).expanduser()
    if not path.is_absolute():
        path=BASE/path
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() not in ['.keras','.h5','.hdf5']:
        raise ValueError('Cần model đầy đủ .keras/.h5; không dùng riêng model.weights.h5.')
    return path.resolve()

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def open_image(raw):
    # Dùng ảnh gốc cho TF decode; PIL chỉ kiểm tra và hiển thị.
    with Image.open(io.BytesIO(raw)) as im:
        if getattr(im,'n_frames',1)!=1 or im.mode in ('I','F') or im.mode.startswith('I;16'):
            raise ValueError('Cần ảnh tĩnh 8-bit; không tự ép ảnh 16-bit/đa frame.')
        im.load()
        image=im.convert('RGB')
        image.info['_original_bytes']=bytes(raw)
        return image

def image_tensor(image,size):
    import tensorflow as tf
    raw=image.info.get('_original_bytes')
    if raw is None:
        raise ValueError('Thiếu bytes ảnh gốc; đọc ảnh qua open_image để khớp TensorFlow.')
    with Image.open(io.BytesIO(raw)) as im:
        if im.format not in ('JPEG','PNG','BMP'):
            buf=io.BytesIO()
            im.convert('RGB').save(buf,format='PNG')
            raw=buf.getvalue()
    x=tf.io.decode_image(raw,channels=3,expand_animations=False)
    x=tf.image.resize(x,size,method='bilinear',antialias=False)
    return tf.cast(x,tf.float32)

def run_prediction(model,image,spec):
    import tensorflow as tf
    started=time.perf_counter()
    shape=model.input_shape
    if isinstance(shape,list) or len(shape)!=4 or shape[-1]!=3:
        raise ValueError('Cần model một input RGB dạng (N,H,W,3).')
    dynamic=spec.get('dynamic_size',[224,224])
    size=tuple(int(shape[i] or dynamic[i-1]) for i in [1,2])
    if any(v<=0 for v in size):
        raise ValueError('Kích thước đầu vào phải dương.')
    x=image_tensor(image,size)
    preview=Image.fromarray(np.clip(x.numpy(),0,255).astype('uint8'))
    method=spec['preprocessing']
    if method=='scale_0_1':
        x=x/255.0
    elif method in ['resnet50','vgg16','vgg19']:
        pre={'resnet50':tf.keras.applications.resnet50.preprocess_input,
             'vgg16':tf.keras.applications.vgg16.preprocess_input,
             'vgg19':tf.keras.applications.vgg19.preprocess_input}[method]
        x=pre(x)
    elif method!='raw_0_255':
        raise ValueError('Chuẩn hóa không được hỗ trợ.')
    before=time.perf_counter()
    raw=np.asarray(model(x[None,...],training=False).numpy())
    finished=time.perf_counter()
    if raw.shape not in [(1,1),(1,2)] or not np.isfinite(raw).all():
        raise ValueError('Model phải trả output hữu hạn (1,1) hoặc (1,2).')
    score=raw[0].astype('float64')
    if spec['output_type']=='logits':
        if score.size==1:
            score=1/(1+np.exp(-np.clip(score,-700,700)))
        else:
            score=np.exp(score-score.max()); score/=score.sum()
    if np.any((score<0)|(score>1)):
        raise ValueError('Điểm không thuộc [0,1]; kiểm tra output_type của model.')
    if score.size==1:
        p=float(score[0]); idx=int(spec['binary_output_index'])
        scores=[1-p,p] if idx==1 else [p,1-p]
    else:
        if not np.isclose(score.sum(),1,atol=1e-4):
            raise ValueError('Hai điểm xác suất không có tổng bằng 1.')
        scores=score.tolist()
    threshold=float(spec['decision_threshold'])
    prediction=int(scores[1]>threshold)  # V4/Keras: bằng ngưỡng thuộc index 0.
    result=dict(model_id=spec['id'],model_name=spec['name'],
                predicted_label=spec['class_names'][prediction],class_names=spec['class_names'],
                scores=scores,raw_output=raw.tolist(),threshold=threshold,decision_rule='score_class_1 > threshold',
                inference_ms=(finished-before)*1000,total_ms=(finished-started)*1000,
                parameters=int(model.count_params()),input_size=list(size),
                preprocessing=method,interpolation=spec['interpolation'],
                binary_output_index=int(spec['binary_output_index']),output_type=spec['output_type'])
    return result,preview