# Hướng dẫn Setup và Chạy GLFC trên Kaggle (Tối ưu Multi-GPU)

Tài liệu này hướng dẫn bạn cách chạy repo `sim-glfc` trên Kaggle bằng GPU (P100, T4, hoặc T4 x2) một cách tự động và hiệu quả nhất.

---

## 1. Chuẩn bị dữ liệu (Kaggle Dataset)
1.  Nén (Zip) thư mục `federated_continual_data` và file `30_test_data.pt` (nếu có).
2.  Lên Kaggle -> **Datasets** -> **New Dataset**.
3.  Upload file Zip và đặt tên (ví dụ: `glfc-data`).
4.  Sau khi tạo xong, dữ liệu của bạn sẽ nằm tại: `/kaggle/input/glfc-data/`

---

## 2. Cấu hình đường dẫn trong Code
Kaggle có cấu trúc thư mục khác với máy cá nhân. Bạn cần sửa lại các dòng lệnh gọi dữ liệu trong Notebook hoặc script:

### Thay đổi trong `fl_main.py`
Tìm đến đoạn khai báo `FederatedTabularDataset` và thay thư mục root:

```python
# Cũ (trên máy tính)
train_dataset = FederatedTabularDataset(client_id=i, root_dir='../federated_continual_data', test_file='../30_test_data.pt')

# Mới (trên Kaggle)
train_dataset = FederatedTabularDataset(
    client_id=i, 
    root_dir='/kaggle/input/glfc-data/federated_continual_data', 
    test_file='/kaggle/input/glfc-data/30_test_data.pt'
)
```

### Thay đổi thư mục Log
Để lưu được kết quả (Log/Model), bạn phải lưu vào thư mục `/kaggle/working/`:
```python
# Trong fl_main.py, tìm output_dir
output_dir = '/kaggle/working/training_log/glfc/seed2021'
os.makedirs(output_dir, exist_ok=True)
```

---

## 3. Lệnh chạy tối ưu trên Kaggle (Sử dụng GPU)
Kaggle cung cấp GPU miễn phí, hãy tận dụng nó bằng cách đổi `--device` từ `-1` sang `0`.

**Lệnh chạy gợi ý (6 Task, 5 Round/Task, 10 Local Epoch):**
```bash
!python /kaggle/working/src/fl_main.py \
    --dataset tabular \
    --model_type cnn \
    --device 0 \
    --tasks_global 5 \
    --epochs_global 30 \
    --epochs_local 10
```

---

## 4. Các bước thực hiện nhanh trên Kaggle Notebook
1.  **Upload toàn bộ thư mục `src`** vào mục `Data` (như một dataset) hoặc dùng lệnh `!git clone` từ GitHub của bạn.
2.  **Thêm Dataset chứa dữ liệu** (đã làm ở bước 1).
3.  **Tạo một ô Code (Cell)** để di chuyển vào thư mục code:
    ```python
    import os
    os.chdir('/kaggle/working/src') # Giả định bạn đã copy code vào đây
    ```
4.  **Cài đặt thư viện (nếu cần):**
    ```python
    !pip install scipy sklearn
    ```
5.  **Chạy huấn luyện:** Thực thi lệnh ở mục 3.

---

## 5. Lưu ý quan trọng
*   **Persistent Storage:** Mọi thứ bạn sửa trong Notebook sẽ mất khi tắt Session, trừ khi bạn lưu (Commit/Save Version).
*   **Working Directory:** Chỉ có thư mục `/kaggle/working/` là có quyền Ghi (Write). Đừng cố gắng tạo file log ở thư mục `/kaggle/input/`.
