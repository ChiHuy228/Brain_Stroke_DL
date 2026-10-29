# BrainStroke Local Web

Hướng dẫn cài đặt và sử dụng ứng dụng thử nghiệm bốn mô hình **CustomCNN, ResNet50, VGG16 và VGG19** trên ảnh CT não 2D. Người dùng có thể chọn một hoặc nhiều mô hình để so sánh kết quả trên cùng một ảnh.
Bộ dữ liệu: lấy từ kaggle link https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset/data
## Mục lục

1. [Chuẩn bị](#1-chuẩn-bị)
2. [Mở dự án trong VSCode](#2-mở-dự-án-trong-vscode)
3. [Tạo môi trường và cài thư viện](#3-tạo-môi-trường-và-cài-thư-viện)
4. [Đặt các mô hình](#4-đặt-các-mô-hình)
5. [Khởi chạy ứng dụng](#5-khởi-chạy-ứng-dụng)
6. [Sử dụng ứng dụng](#6-sử-dụng-ứng-dụng)
7. [Dừng và mở lại ứng dụng](#7-dừng-và-mở-lại-ứng-dụng)
8. [Mô tả thao tác trên giao diện](#8-mô-tả-thao-tác-trên-giao-diện)

## 1. Chuẩn bị

Chuẩn bị các thành phần sau:

- **Visual Studio Code:** dùng để mở thư mục dự án và chạy lệnh.
- **Miniconda:** dùng để tạo môi trường Python riêng cho ứng dụng.
- **Mã nguồn ứng dụng**, tệp `requirements.txt` và bốn mô hình `.keras` đã huấn luyện.
- Kết nối Internet khi cài đặt thư viện lần đầu.

### Cài Visual Studio Code

Tải bộ cài phù hợp với hệ điều hành tại [code.visualstudio.com](https://code.visualstudio.com/), sau đó cài đặt và mở VSCode.

### Cài Miniconda

Tải Miniconda theo hệ điều hành tại [trang tải chính thức của Anaconda](https://www.anaconda.com/download). Xem [hướng dẫn cài Conda chính thức](https://docs.conda.io/projects/conda/en/stable/user-guide/install/) để chọn bộ cài phù hợp.

- **Windows:** chạy bộ cài, hoàn thành các bước trên màn hình; sau đó mở **Anaconda Prompt (Miniconda3)** từ Start Menu.
- **Linux:** tải bộ cài Miniconda dạng `.sh`, mở Terminal tại thư mục chứa bộ cài, gõ `bash ` rồi kéo tệp bộ cài vào Terminal và nhấn Enter. Làm theo hướng dẫn trên màn hình, cho phép khởi tạo shell khi được hỏi, rồi đóng và mở lại Terminal.

Kiểm tra sau khi cài:

```bash
conda --version
```

Các bước tiếp theo dùng **Python 3.11**, phù hợp với hướng dẫn Python 3.11 hoặc 3.12 trong `requirements.txt` của dự án.

## 2. Mở dự án trong VSCode

1. Giải nén gói mã nguồn thành thư mục `BrainStroke_Local_Web`.
2. Mở VSCode, chọn **File → Open Folder**.
3. Chọn thư mục chứa `app.py`, `inference.py`, `model_config.json` và `requirements.txt`.
4. Chọn **Terminal → New Terminal** để mở cửa sổ nhập lệnh.

Trên Windows, có thể thực hiện các lệnh Conda bên dưới trong **Anaconda Prompt (Miniconda3)**. Gõ `cd /d `, thêm đường dẫn thư mục dự án trong dấu ngoặc kép rồi nhấn Enter. Trên Linux, dùng Terminal đã nhận lệnh Conda.

Ví dụ chuyển thư mục bằng Terminal Linux:

```bash
cd "/duong/dan/BrainStroke_Local_Web"
```

Ví dụ trong Anaconda Prompt trên Windows:

```bat
cd /d "D:\BrainStroke_Local_Web"
```

Thay đường dẫn ví dụ bằng vị trí thực tế của thư mục đã giải nén. Mọi lệnh cài đặt và chạy ứng dụng dưới đây được thực hiện tại thư mục chứa `app.py`.

## 3. Tạo môi trường và cài thư viện

### Bước 1 — Tạo môi trường Python

Chạy lần lượt:

```bash
conda create -n brainstroke python=3.11 pip -y
conda activate brainstroke
```

`brainstroke` là tên môi trường của dự án. Bước tạo môi trường chỉ cần thực hiện một lần; nếu đã có môi trường này thì chỉ chạy lệnh kích hoạt.

### Bước 2 — Cài thư viện

```bash
python -m pip install -r requirements.txt
```

Đợi lệnh hoàn tất trước khi chuyển sang bước tiếp theo. Nội dung thư viện được lấy từ tệp `requirements.txt` đi kèm:

| Thư viện | Phiên bản yêu cầu |
|---|---|
| Streamlit | `==1.50.0` |
| TensorFlow | `==2.20.0` |
| NumPy | `>=1.26,<2.3` |
| pandas | `>=2.2,<3` |
| Pillow | `>=10.4,<12` |
| h5py | `>=3.11,<4` |

### Bước 3 — Chọn Python trong VSCode

1. Mở **Extensions**, cài tiện ích **Python** của Microsoft.
2. Nhấn **Ctrl + Shift + P**.
3. Chọn **Python: Select Interpreter**.
4. Chọn Python thuộc môi trường **brainstroke**.

Việc chọn interpreter giúp VSCode sử dụng đúng môi trường của dự án. Trong mỗi Terminal mới dùng để chạy web, kích hoạt môi trường bằng `conda activate brainstroke`.

## 4. Đặt các mô hình

Tạo thư mục `models` cạnh `app.py`, đặt nguyên bốn tệp mô hình vào bên trong:

```text
BrainStroke_Local_Web/
├── app.py
├── inference.py
├── model_config.json
├── requirements.txt
├── README.md
└── models/
    ├── CustomCNN_best.keras
    ├── ResNet50_best.keras
    ├── VGG16_best.keras
    └── VGG19_best.keras
```

- Lấy các tệp `.keras` đã xuất từ quá trình huấn luyện trên Kaggle.
- Nếu tải cả gói kết quả `.zip`, giải nén gói để lấy các tệp `.keras`.
- Giữ nguyên các tệp `.keras`, không giải nén nội dung bên trong từng mô hình.
- Giữ đúng tên và chữ hoa/chữ thường như cấu hình. Nếu dùng tên khác, cập nhật đường dẫn tương ứng trong `model_config.json`.
- Dùng mô hình và cấu hình cùng phiên bản huấn luyện.

Hướng dẫn này chạy trực tiếp `app.py`; không cần `run_app.py`, tệp `.sh` hoặc `.bat` để thực hiện các bước bên dưới.

## 5. Khởi chạy ứng dụng

Trong Terminal tại thư mục dự án, chạy:

```bash
conda activate brainstroke
python -m streamlit run app.py 
```

Khi Terminal hiển thị **Local URL**, mở địa chỉ đó trong trình duyệt. Với cổng mặc định:

**[http://localhost:8501](http://localhost:8501)**

Giữ Terminal mở trong suốt thời gian sử dụng. Ứng dụng nạp các mô hình đã huấn luyện để dự đoán; thao tác này không huấn luyện lại mô hình.

## 6. Sử dụng ứng dụng

### Bước 1 — Chọn mô hình

Trong phần lựa chọn mô hình, chọn một hoặc nhiều mạng muốn thử nghiệm. Có thể chọn cả bốn để đối chiếu trên cùng một ảnh.

### Bước 2 — Tải ảnh

Chọn một ảnh CT não 2D, ưu tiên ảnh PNG/JPG/JPEG/BMP 8-bit thuộc danh sách test đã chia trong notebook. Kiểm tra ảnh xem trước để xác nhận đúng ảnh muốn thử nghiệm.

### Bước 3 — Chạy phân tích

Nhấn **Phân tích ảnh** và đợi các mô hình được chọn hoàn thành dự đoán.

### Bước 4 — Xem và so sánh kết quả

Đọc kết quả riêng của từng mô hình:

| Thông tin | Ý nghĩa |
|---|---|
| Nhãn Normal/Stroke | Nhãn dự đoán của mô hình cho ảnh được tải lên. |
| Điểm đầu ra | Điểm của các lớp theo đầu ra mô hình. |
| Ngưỡng | Mốc được sử dụng để chuyển điểm thành nhãn. |
| Thời gian suy luận | Thời gian đo trong lượt chạy của từng mô hình. |
| Bảng so sánh | Tổng hợp kết quả của những mô hình đã chọn. |

Điểm của một ảnh không phải Accuracy trên toàn bộ tập test. Chọn nhiều mô hình sẽ hiển thị các dự đoán riêng để so sánh.

### Bước 5 — Lưu kết quả

Sử dụng nút tải **JSON** hoặc **CSV** trên giao diện để lưu kết quả của lượt thử nghiệm. Khi đổi ảnh hoặc thay danh sách mô hình, nhấn **Phân tích ảnh** lại để cập nhật kết quả.

Ứng dụng phục vụ nghiên cứu và thử nghiệm phân loại ảnh CT 2D; kết quả không thay thế chẩn đoán của bác sĩ.

## 7. Dừng và mở lại ứng dụng

### Dừng ứng dụng

Quay lại Terminal đang chạy web và nhấn **Ctrl + C**.

### Mở lại vào lần sử dụng tiếp theo

Mở thư mục dự án trong VSCode, mở Terminal rồi chạy:

```bash
conda activate brainstroke
python -m streamlit run app.py 
```

Mở Local URL được in trong Terminal. Không cần tạo môi trường hoặc cài lại thư viện mỗi lần sử dụng.

## 8. Mô tả thao tác trên giao diện

Phần này dành để bổ sung ảnh chụp thực tế và hướng dẫn trực quan. Tạo thư mục `docs/images` trong dự án, lưu ảnh theo các tên bên dưới rồi thay dòng **Vị trí ảnh** bằng cú pháp Markdown tương ứng.

### 8.1 Giao diện chính

**Mô tả:** giới thiệu bố cục trang, khu vực lựa chọn mô hình và khu vực tải ảnh.

**Vị trí ảnh:** `docs/images/01-giao-dien-chinh.png`

```markdown
![Giao diện chính](docs/images/01-giao-dien-chinh.png)
```

### 8.2 Chọn một hoặc nhiều mô hình

**Mô tả:** minh họa thao tác chọn một mạng hoặc chọn nhiều mạng để so sánh.

**Vị trí ảnh:** `docs/images/02-chon-mo-hinh.png`

```markdown
![Chọn mô hình](docs/images/02-chon-mo-hinh.png)
```

### 8.3 Tải ảnh và chạy phân tích

**Mô tả:** minh họa vị trí tải ảnh, ảnh xem trước và nút **Phân tích ảnh**.

**Vị trí ảnh:** `docs/images/03-tai-anh-phan-tich.png`

```markdown
![Tải ảnh và phân tích](docs/images/03-tai-anh-phan-tich.png)
```

### 8.4 Xem kết quả và bảng so sánh

**Mô tả:** giải thích nhãn dự đoán, điểm đầu ra và thời gian suy luận của các mô hình trên cùng ảnh. Dùng kết quả từ lượt chạy thực tế khi chụp hình.

**Vị trí ảnh:** `docs/images/04-ket-qua-so-sanh.png`

```markdown
![Kết quả và bảng so sánh](docs/images/04-ket-qua-so-sanh.png)
```

### 8.5 Tải báo cáo kết quả

**Mô tả:** minh họa thao tác tải JSON/CSV để lưu kết quả thử nghiệm.

**Vị trí ảnh:** `docs/images/05-tai-ket-qua.png`

```markdown
![Tải báo cáo kết quả](docs/images/05-tai-ket-qua.png)
```
