"""Giao diện web local để so sánh các model phân loại 2D có sẵn."""
import copy
import csv
import hashlib
import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from inference import BASE, PREPROCESS, RESIZE, load_config, open_image, resolve_model, run_prediction, sha256_file

st.set_page_config(page_title="Stroke Lab · Test model",page_icon="🧠",layout="wide")
st.markdown("""<style>
.stApp{background:#f5f7fb;color:#142b45}
.block-container{max-width:1360px;padding-top:2rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e1e8f0}
.hero{background:linear-gradient(115deg,#102b4d,#174d70);padding:30px 34px;border-radius:20px;color:white;margin-bottom:24px}
.hero h1{color:white;font-size:2.3rem;margin:5px 0 10px}.hero p{color:#dce9f3;margin:0;max-width:840px}
.eyebrow{letter-spacing:.16em;color:#72d6d1;font-size:.78rem;font-weight:700}
[data-testid="stMetric"]{background:white;border:1px solid #e1e8f0;border-radius:14px;padding:14px}
.stButton>button[kind="primary"]{background:#116f78;border-color:#116f78;border-radius:10px}
.note{font-size:.88rem;color:#5b6c80}
</style>""",unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="eyebrow">STROKE LAB / LOCAL MODEL TESTING</div>
<h1>Một ảnh. Bốn mô hình. Kết quả để đối chiếu.</h1>
<p>Tải ảnh 2D, chạy Custom CNN, ResNet50, VGG16 hoặc VGG19 và xem điểm từng lớp trên cùng một giao diện.</p></div>""",unsafe_allow_html=True)
st.caption("Bản thử nghiệm nghiên cứu • Điểm đầu ra chưa được hiệu chỉnh thành xác suất lâm sàng • Không xác nhận hoặc loại trừ đột quỵ.")

try:
    config = load_config()
except Exception as error:
    st.error(f"Không đọc được model_config.json: {error}"); st.stop()
specs = {m["id"]:m for m in config["models"]}

@st.cache_resource(show_spinner=False)
def load_model_cached(path, size, modified_ns):
    # Chỉ nạp model được chọn; không tải pretrained weights hoặc train lại.
    import tensorflow as tf
    model = tf.keras.models.load_model(path,compile=False)
    return model, threading.Lock()

@st.cache_data(show_spinner=False)
def model_digest(path,size,modified_ns):
    return sha256_file(path)

with st.sidebar:
    st.title("🧠 Stroke Lab")
    st.caption("Không gian thử nghiệm local")
    selected = st.multiselect("Mô hình cần chạy",list(specs),default=list(specs),format_func=lambda k:specs[k]["name"])
    st.divider()
    st.subheader("File model")
    availability = {}
    for mid,spec in specs.items():
        try:
            path = resolve_model(spec); availability[mid] = path
            st.success(f"{spec['name']}: {path.name}")
        except FileNotFoundError:
            st.caption(f"○ {spec['name']} — chưa có file")
    st.caption("Đặt model trong models/ hoặc sửa path trong model_config.json. Hỗ trợ model đầy đủ .h5 và .keras.")
    st.divider()
    override = st.checkbox("Điều chỉnh cấu hình cho lượt này",value=False)
    active_specs = copy.deepcopy(specs)
    if override:
        for mid in selected:
            spec = active_specs[mid]
            with st.expander(spec["name"]):
                methods = list(PREPROCESS)
                spec["preprocessing"] = st.selectbox("Chuẩn hóa",methods,index=methods.index(spec["preprocessing"]),format_func=lambda k:PREPROCESS[k],key=f"prep_{mid}")
                options = list(RESIZE)
                spec["interpolation"] = st.selectbox("Nội suy resize",options,index=options.index(spec["interpolation"]),key=f"resize_{mid}")
                spec["class_names"] = [st.text_input("Tên lớp index 0",spec["class_names"][0],key=f"label0_{mid}"),st.text_input("Tên lớp index 1",spec["class_names"][1],key=f"label1_{mid}")]
                spec["binary_output_index"] = st.selectbox("Nếu đầu ra 1 số: số này thuộc lớp index nào?",[0,1],index=spec["binary_output_index"],key=f"index_{mid}")
                spec["output_type"] = st.selectbox("Kiểu đầu ra",["probabilities","logits"],index=0 if spec["output_type"]=="probabilities" else 1,key=f"type_{mid}")
                spec["decision_threshold"] = st.number_input("Ngưỡng điểm lớp index 1",min_value=0.01,max_value=0.99,value=float(spec["decision_threshold"]),step=.01,key=f"threshold_{mid}")
        st.caption("Đổi cấu hình phải dựa vào notebook train; không chọn cách cho ra nhãn mong muốn. Thay đổi tại đây không ghi đè file JSON.")

left,right = st.columns([1,1.3],gap="large")
with left:
    st.subheader("01 / Ảnh đầu vào")
    upload = st.file_uploader("Chọn ảnh để kiểm tra",type=["png","jpg","jpeg","bmp","tif","tiff","webp"],help="Một ảnh 2D 8-bit đúng cách xuất ảnh khi training. Không nhận .nii.gz/DICOM trực tiếp.")
    image = None
    image_bytes = None
    if upload:
        try:
            image_bytes = upload.getvalue(); image = open_image(image_bytes)
            st.image(image,caption=upload.name,width="stretch")
            a,b,c = st.columns(3)
            a.metric("Chiều rộng",f"{image.width:,} px"); b.metric("Chiều cao",f"{image.height:,} px"); c.metric("Dung lượng",f"{len(image_bytes)/1024:.0f} KB")
        except (UnidentifiedImageError,ValueError,OSError,Image.DecompressionBombError) as error:
            st.error(f"Không đọc được ảnh: {error}")
    else:
        st.info("Chọn ảnh PNG/JPG đã xử lý giống ảnh đầu vào của mô hình.")
    st.caption("Ảnh được xử lý trong bộ nhớ của ứng dụng local. Code không gửi ảnh hoặc model tới dịch vụ bên ngoài.")

with right:
    st.subheader("02 / Chạy và so sánh")
    st.info("Đã đối chiếu notebook training bạn gửi: đầu vào RGB 0–255, resize TensorFlow bilinear, Normal=0 và Stroke=1. Chuẩn hóa nằm trong model, không chia thêm /255 ở web. Cấu hình hỗ trợ notebook V4; cần đặt đúng các file model đã tải về. Điểm hiển thị chỉ được tính khi chạy model thật.")
    available_selected = [mid for mid in selected if mid in availability]
    if selected and len(available_selected)<len(selected):
        st.warning("Một số model chưa có file. Ứng dụng sẽ chỉ chạy các model hiện có.")
    active_key = hashlib.sha256(json.dumps({k:active_specs[k] for k in selected},sort_keys=True).encode()).hexdigest()
    file_versions = [(mid, str(availability[mid]), availability[mid].stat().st_size,availability[mid].stat().st_mtime_ns) for mid in available_selected]
    request_key = (hashlib.sha256(image_bytes).hexdigest() if image_bytes else None,active_key,tuple(file_versions))
    if st.button("Phân tích ảnh",type="primary",width="stretch",disabled=image is None or not available_selected):
        results,errors,previews = [],[],{}
        with st.status("Đang nạp và chạy mô hình…",expanded=True) as status:
            for mid in available_selected:
                spec = active_specs[mid]; path = availability[mid]; stat = path.stat()
                st.write("Đang chạy " + spec["name"])
                try:
                    if len(set(spec["class_names"]))!=2 or not all(spec["class_names"]):
                        raise ValueError("Cần hai tên lớp khác nhau, không để trống.")
                    model,lock = load_model_cached(str(path),stat.st_size,stat.st_mtime_ns)
                    with lock:
                        result,preview = run_prediction(model,image,spec)
                    result["model_file"] = path.name
                    result["model_sha256"] = model_digest(str(path),stat.st_size,stat.st_mtime_ns)
                    results.append(result); previews[mid] = preview
                except Exception as error:
                    errors.append(dict(model=spec["name"],error=f"{type(error).__name__}: {error}"))
            status.update(label=f"Hoàn tất {len(results)}/{len(available_selected)} mô hình",state="complete" if results else "error",expanded=False)
        import tensorflow as tf
        st.session_state["prediction"] = dict(key=request_key,results=results,errors=errors,previews=previews,
                file_name=upload.name,image_sha256=request_key[0],created_utc=datetime.now(timezone.utc).isoformat(),
                versions={"tensorflow":tf.__version__,"streamlit":st.__version__})
    saved = st.session_state.get("prediction")
    if saved and saved["key"] == request_key:
        for error in saved["errors"]:
            st.error(error["model"] + ": " + error["error"])
            st.caption("Nếu lỗi tương thích Keras/custom layer: dùng đúng môi trường train và mã lớp tùy chỉnh. Không tự tắt safe mode hoặc đổi kiến trúc.")
        for result in saved["results"]:
            with st.container(border=True):
                st.markdown(f"**{result['model_name']}**")
                st.write("Nhãn dự đoán: **" + result["predicted_label"] + "**")
                for label,score in zip(result["class_names"],result["scores"]):
                    st.progress(float(score),text=f"{label}: {score*100:.2f}% — điểm mô hình")
                st.caption(f"Ngưỡng: {result['threshold']:.6g} · Suy luận: {result['inference_ms']:.1f} ms · Tổng tiền xử lý + suy luận: {result['total_ms']:.1f} ms")
        if saved["results"]:
            st.caption("Thời gian trên không gồm nạp model/hash file; lần gọi đầu có thể chậm hơn do khởi tạo runtime.")
    elif saved:
        st.caption("Ảnh, model hoặc cấu hình đã đổi. Bấm Phân tích ảnh để có kết quả mới.")
    else:
        st.caption("Kết quả chỉ xuất hiện sau khi chạy model thật. Không có điểm Accuracy đặt sẵn.")

st.divider()
tab1,tab2,tab3 = st.tabs(["Bảng so sánh & tải kết quả","Cấu hình xử lý ảnh","Hướng dẫn sử dụng"])
with tab1:
    saved = st.session_state.get("prediction")
    if saved and saved["key"]==request_key and saved["results"]:
        rows = []
        for r in saved["results"]:
            rows.append({"Mô hình":r["model_name"],"Nhãn dự đoán":r["predicted_label"],
                         "Lớp 0":r["class_names"][0],"Điểm lớp 0":r["scores"][0],
                         "Lớp 1":r["class_names"][1],"Điểm lớp 1":r["scores"][1],
                         "Ngưỡng":r["threshold"],"Suy luận (ms)":r["inference_ms"],"Số tham số":r["parameters"]})
        table = pd.DataFrame(rows); st.dataframe(table,hide_index=True,width="stretch")
        export = {k:saved[k] for k in ["file_name","image_sha256","created_utc","versions","results","errors"]}
        a,b = st.columns(2)
        a.download_button("Tải báo cáo JSON",json.dumps(export,ensure_ascii=False,indent=2),file_name="prediction_report.json",mime="application/json")
        b.download_button("Tải bảng CSV",table.to_csv(index=False).encode("utf-8-sig"),file_name="prediction_comparison.csv",mime="text/csv")
        with st.expander("Ảnh sau resize và đầu ra thô"):
            for r in saved["results"]:
                st.image(saved["previews"][r["model_id"]],caption=r["model_name"]+" — ảnh trước chuẩn hóa pixel",width=224)
                st.json(r)
    else:
        st.write("Chưa có kết quả cho ảnh và cấu hình hiện tại.")
with tab2:
    st.caption("Kích thước cố định đọc từ model. Chỉ dùng dynamic_size khi model có chiều H/W chưa xác định.")
    st.json({"configuration_source":config["configuration_source"],"models":[active_specs[k] for k in selected]})
with tab3:
    st.markdown("""1. Đặt đủ CustomCNN_best.keras, ResNet50_best.keras, VGG16_best.keras, VGG19_best.keras trong `models/`.
2. Dùng model_config.json xuất từ notebook V4; ứng dụng cũng đọc được cấu trúc config web cũ.
3. Đối chiếu `class_indices`, chuẩn hóa pixel, nội suy và ngưỡng với notebook training.
4. Chọn một hoặc nhiều model, tải một ảnh 2D đúng loại dữ liệu đã train, bấm **Phân tích ảnh**.
5. Tải JSON/CSV để lưu điểm đầu ra, cấu hình, thời gian đo và hash model.

Đây là kiểm tra suy luận từng ảnh, chưa phải đánh giá Accuracy/F1 của tập test. Không có nhãn thật thì chưa kết luận model đúng/sai. Không gộp điểm các model thành một chẩn đoán hoặc tự chọn model có điểm cao nhất làm model tốt nhất.

Các model Keras phân loại 2D này khác model phân đoạn APIS/ISLES `.pt`. Ứng dụng không tự biến ảnh NIfTI/DICOM thành ảnh đầu vào hoặc phân đoạn tổn thương.""")