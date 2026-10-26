# AGENTS.md — Hướng dẫn cộng tác dự án

## Mục tiêu dự án

Dự án học thuật xây dựng mô hình đánh giá rủi ro tín dụng, dùng tối thiểu hai bộ dữ liệu và yêu cầu phân tích kết quả, lỗi mô hình. Đây là codebase phục vụ môn Học máy; không dùng đầu ra để ra quyết định tín dụng thực tế.

## Cấu trúc mã nguồn

- `src/data/`: tải, đọc và tiền xử lý dữ liệu.
- `src/features/`: tạo và lựa chọn đặc trưng.
- `src/models/`: huấn luyện và suy luận.
- `src/evaluation/`: metric và phân tích lỗi.
- `src/visualization/`: biểu đồ EDA và đánh giá.
- `configs/`: cấu hình YAML.
- `notebooks/`: EDA, modeling và evaluation theo thứ tự.
- `tests/`: kiểm thử tự động.
- `reports/`: báo cáo và biểu đồ sinh ra.

## Quy tắc làm việc với dữ liệu và ML

- Không commit dữ liệu thô/đã xử lý, mô hình sinh ra, credentials hoặc file môi trường.
- Giữ riêng từng bộ dữ liệu trong huấn luyện và đánh giá; không gộp nhãn/đặc trưng khác định nghĩa thành một tập chung nếu chưa có thiết kế và lý do rõ ràng.
- Chia train/test trước khi fit các phép biến đổi học tham số. Fit imputer, encoder, scaler, chọn đặc trưng và ngưỡng chỉ trên train; resampling chỉ áp dụng trong train/fold train. Giữ nguyên phân bố test.
- Không dùng test set để chọn mô hình, tuning siêu tham số hoặc chọn ngưỡng.
- Với phân lớp mất cân bằng, báo cáo metric phù hợp (ví dụ Recall, Precision, F1, ROC-AUC, Average Precision) và confusion matrix; giải thích chi phí FP/FN.
- Phân biệt rõ kết quả dữ liệu thật với dữ liệu tổng hợp. Ghi nguồn và giới hạn sử dụng của dữ liệu.
- SHAP là công cụ giải thích, không phải bằng chứng về tính công bằng hoặc quan hệ nhân quả.

## Quy ước mã nguồn

- Dùng Python, theo phong cách và tên biến nhất quán với mã hiện có.
- Ưu tiên hàm nhỏ có kiểu dữ liệu/giá trị đầu vào rõ ràng; ghi docstring khi cần giải thích mục đích hoặc tham số không hiển nhiên.
- Đọc cấu hình từ `configs/config.yaml` thay vì rải tham số cố định trong pipeline.
- Không thêm dependency nếu thư viện chuẩn hoặc dependency hiện có đáp ứng được.
- Cập nhật test và README khi thay đổi hành vi, giao diện lệnh hoặc cách thiết lập dự án.

## Kiểm tra trước khi gửi thay đổi

Chạy các kiểm tra phù hợp với thay đổi:

```bash
python -m compileall -q main.py src tests
pytest -q
 git diff --check
```

Nếu thiếu dependency hoặc dữ liệu để chạy một kiểm tra, ghi rõ lệnh chưa chạy được và nguyên nhân trong PR; không mô tả kiểm tra chưa chạy là thành công.

## Git workflow

- Không commit trực tiếp lên `main`; tạo nhánh `feat/<ten>`, `fix/<ten>`, `docs/<ten>` hoặc `test/<ten>` từ `main` đã cập nhật.
- Commit theo Conventional Commits, ví dụ `docs: add contributor guidance`.
- Mỗi PR nên có mục tiêu, thay đổi chính, kiểm tra đã chạy và giới hạn còn lại.
- Yêu cầu ít nhất một thành viên review trước khi merge; xử lý góp ý và kiểm tra lại.
- Không force-push nhánh dùng chung nếu chưa thống nhất.
- Xóa nhánh sau khi merge nếu nhóm không cần giữ.
