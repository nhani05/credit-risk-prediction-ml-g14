# Mô hình đánh giá rủi ro tín dụng

Dự án bài tập lớn môn **Học máy** của Nhóm 14. Mục tiêu là xây dựng và so sánh các mô hình phân loại khách hàng theo rủi ro tín dụng trên **ít nhất hai bộ dữ liệu**, phân tích kết quả và lỗi, đồng thời thử nghiệm một số kỹ thuật ngoài nội dung cơ bản như Gradient Boosting, Stacking và SHAP.

> Đây là dự án học thuật, không phải hệ thống thẩm định tín dụng dùng trong thực tế. Kết quả trên dữ liệu công khai không nên được dùng để đưa ra quyết định tài chính hoặc quyết định đối với cá nhân.

## Mục tiêu học tập

- Thực hành quy trình ML có thể tái lập: khám phá dữ liệu, tiền xử lý, xây dựng đặc trưng, huấn luyện, đánh giá và phân tích lỗi.
- So sánh các mô hình cơ sở và mô hình ensemble trên hai tập dữ liệu riêng biệt.
- Đánh giá trong bối cảnh mất cân bằng lớp, không chỉ dựa vào Accuracy.
- Phân tích False Positive/False Negative, đường cong ROC và Precision–Recall, hiệu chỉnh xác suất và khả năng giải thích mô hình.

## Bộ dữ liệu

### 1. German Credit (Statlog)

- Nguồn: [UCI Machine Learning Repository – Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- Có 1.000 quan sát, 20 biến đầu vào và nhãn phân loại.
- File được tải bởi script dùng mã hóa nhãn UCI: `1` (good credit) được chuyển thành `0`, `2` (bad credit) được chuyển thành `1`.

### 2. Give Me Some Credit

- Nguồn: [Kaggle – Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
- Nhãn `SeriousDlqin2yrs` biểu thị khách hàng có gặp tình trạng quá hạn nghiêm trọng trong hai năm hay không.
- Kaggle yêu cầu tài khoản và chấp thuận điều khoản cuộc thi. Hãy tải `cs-training.csv` từ Kaggle và đặt trong `data/raw/`. Có thể thiết lập Kaggle API theo tài liệu Kaggle để tải tự động.

**Quan trọng:** Không gộp hai tập dữ liệu thành một tập huấn luyện duy nhất. Chúng có khác biệt về định nghĩa nhãn, thời điểm, quần thể và đặc trưng; hãy huấn luyện/đánh giá riêng rồi thảo luận về tính khái quát hóa và giới hạn so sánh. Kiểm tra điều khoản sử dụng của từng nguồn trước khi phân phối dữ liệu. Dữ liệu không được commit vào Git.

## Cấu trúc dự án

```text
.
├── configs/
│   └── config.yaml              # Tham số tiền xử lý, mô hình, đầu ra
├── data/
│   ├── raw/                     # Dữ liệu nguồn (không commit)
│   ├── processed/               # Dữ liệu đã xử lý (không commit)
│   └── external/                # Dữ liệu ngoài (không commit)
├── notebooks/
│   ├── 01_EDA.ipynb             # Khám phá dữ liệu
│   ├── 02_Modeling.ipynb        # Tiền xử lý và huấn luyện
│   └── 03_Evaluation.ipynb      # Đánh giá và phân tích lỗi
├── reports/
│   └── figures/                 # Biểu đồ được tạo bởi pipeline
├── src/
│   ├── data/                    # Tải, nạp, tiền xử lý dữ liệu
│   ├── features/                # Tạo/chọn đặc trưng và PCA
│   ├── models/                  # Huấn luyện, lưu và dự đoán
│   ├── evaluation/              # Metrics và phân tích lỗi
│   ├── visualization/           # Biểu đồ EDA và đánh giá
│   └── config.py                # Đọc YAML config
├── tests/                       # Kiểm thử
├── main.py                      # Điểm chạy pipeline
├── requirements.txt
└── README.md
```

## Cài đặt

Khuyến nghị Python 3.10 hoặc 3.11 và môi trường ảo riêng.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chỉ cần tải German Credit, chạy:

```bash
python -m src.data.download_data
```

Đặt `cs-training.csv` tải từ Kaggle vào `data/raw/`. Cấu trúc thư mục mong đợi:

```text
data/raw/german_credit.csv
data/raw/cs-training.csv
```

## Chạy dự án

Từ thư mục gốc repository:

```bash
# Chạy pipeline mặc định trên các bộ dữ liệu tìm thấy
python main.py

# Chạy pipeline cho một bộ dữ liệu
python main.py --dataset german_credit
python main.py --dataset give_me_credit

# Bật tìm kiếm siêu tham số (có thể chạy lâu)
python main.py --dataset german_credit --tune

# Bật phân tích SHAP (có thể chạy lâu)
python main.py --dataset german_credit --shap

# Chọn file cấu hình khác
python main.py --config configs/config.yaml
```

Hiện tại `--step download` dùng để tải dữ liệu. Các bước `preprocess`, `train`, `evaluate`, `visualize` chưa được hỗ trợ như lệnh độc lập và sẽ báo lỗi rõ ràng; chạy pipeline bằng `--step all` hoặc dùng các notebook để thử từng giai đoạn. Kết quả được ghi vào `models/`, `data/processed/` và `reports/` (các đầu ra này được loại khỏi Git).

### Chạy notebook

```bash
jupyter notebook
```

Mở lần lượt `01_EDA.ipynb`, `02_Modeling.ipynb` và `03_Evaluation.ipynb`. Chạy notebook từ thư mục gốc để các đường dẫn tương đối và import `src` hoạt động như dự kiến. Các notebook có thể cần thời gian đáng kể khi chạy cross-validation, ensemble, SMOTE hoặc SHAP.

## Phương pháp

### Tiền xử lý và đặc trưng

- Điền giá trị thiếu bằng median cho dữ liệu số và giá trị phổ biến cho dữ liệu phân loại.
- Capping ngoại lai theo IQR; mã hóa biến phân loại; chuẩn hóa bằng StandardScaler hoặc MinMaxScaler.
- Chia train/test có phân tầng và xử lý mất cân bằng bằng SMOTE/undersampling trên dữ liệu huấn luyện.
- Tạo một số đặc trưng nghiệp vụ, chọn đặc trưng bằng Mutual Information, Chi-square hoặc RFE; hỗ trợ PCA.

Trong báo cáo cuối cùng, cần mô tả rõ các bước đã thực sự chạy và tham số dùng. Với đánh giá nghiên cứu nghiêm ngặt, mọi bước học tham số từ dữ liệu (imputation, encoding, scaling, chọn đặc trưng, xử lý mất cân bằng) phải được fit **chỉ trên tập train**, tốt nhất được đóng gói trong `Pipeline`/`ColumnTransformer` của scikit-learn. Không dùng tập test để chọn mô hình, ngưỡng hoặc siêu tham số. Kiểm tra pipeline trước khi trích dẫn chỉ số cuối cùng.

### Mô hình

- Baseline: Logistic Regression, Decision Tree, K-Nearest Neighbors, SVM.
- Ensemble: Random Forest, Gradient Boosting, XGBoost, LightGBM và Stacking.
- Tùy chọn GridSearchCV/RandomizedSearchCV và Stratified K-fold cross-validation.

XGBoost và LightGBM được nạp tùy chọn nếu đã cài. Với cài đặt tối thiểu hoặc môi trường không hỗ trợ một thư viện, kiểm tra log để biết mô hình nào được bỏ qua.

### Đánh giá và phân tích lỗi

Báo cáo Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision, MCC, Brier score và confusion matrix. Trong bài toán tín dụng mất cân bằng, nên giải thích trade-off Precision–Recall và chi phí tương đối của FN (bỏ lọt khách hàng rủi ro) so với FP (đánh dấu nhầm khách hàng tốt). Chọn metric và ngưỡng dựa trên mục tiêu được nêu trước, không chọn dựa trên tập test.

### Nội dung mở rộng

SHAP (SHapley Additive exPlanations) được cung cấp để diễn giải mức đóng góp đặc trưng vào dự đoán. Đây là công cụ giải thích mô hình, không chứng minh mô hình công bằng hoặc quan hệ nhân quả. Khi sử dụng, thảo luận thêm về quyền riêng tư, thiên lệch dữ liệu, khả năng giải thích và giới hạn của dữ liệu đại diện.

## Kiểm thử

Sau khi cài thư viện, chạy:

```bash
pytest -q
```

## Tái lập và báo cáo

- Cố định `random_state` và ghi lại phiên bản Python/thư viện, nguồn dữ liệu, ngày tải và cấu hình.
- Không so sánh trực tiếp điểm số giữa hai bộ dữ liệu như thể chúng là cùng một bài toán.
- Báo cáo kết quả của từng bộ dữ liệu riêng, gồm phân bố nhãn, phương pháp chia dữ liệu, metric phù hợp, confusion matrix và phân tích lỗi.
- Lưu cấu hình/siêu tham số và chỉ dùng test set để đánh giá cuối cùng một lần sau khi đã chốt quy trình.
- Phân biệt kết quả trên dữ liệu thật với dữ liệu demo/tổng hợp; không trình bày dữ liệu giả lập như dữ liệu Kaggle.

## Giới hạn

Đây là codebase học tập ban đầu, cần được rà soát và kiểm thử trong môi trường của nhóm trước khi dùng làm kết quả nộp. Các bộ dữ liệu công khai có thể đã cũ, khác phân bố thực tế, có thiên lệch và không đại diện cho quy trình cấp tín dụng hiện hành. Mô hình dự đoán không thay thế thẩm định chuyên môn hay đánh giá tác động công bằng.

## Phân công công việc (nhóm 3 thành viên)

> Thay “Thành viên 1/2/3” bằng tên thành viên thực tế. Các phần được chia theo trách nhiệm chính; cả nhóm cùng thống nhất quy trình, review mã nguồn và chuẩn bị báo cáo/thuyết trình.

| Thành viên | Phạm vi phụ trách | Công việc và đầu ra cụ thể |
|---|---|---|
| **Thành viên 1 — Dữ liệu & EDA** | Nguồn dữ liệu, kiểm tra dữ liệu, khám phá dữ liệu và tiền xử lý | Tìm hiểu điều khoản và tải tối thiểu hai bộ dữ liệu; ghi nguồn, mô tả biến, nhãn và giới hạn dữ liệu. Kiểm tra kiểu dữ liệu, giá trị thiếu, ngoại lai, trùng lặp, phân bố nhãn và tương quan. Phụ trách `src/data/`, notebook `01_EDA.ipynb` và phần EDA trong báo cáo. Bàn giao dữ liệu đã kiểm tra cùng các quyết định tiền xử lý có giải thích. |
| **Thành viên 2 — Đặc trưng & mô hình** | Feature engineering, baseline, huấn luyện và tinh chỉnh | Thiết kế đặc trưng mới phù hợp với từng bộ dữ liệu; xây dựng baseline Logistic Regression/Decision Tree và các mô hình so sánh như Random Forest, Gradient Boosting, SVM hoặc KNN. Thử nghiệm lựa chọn đặc trưng, cross-validation và tuning trong phạm vi thời gian cho phép. Phụ trách `src/features/`, `src/models/` và notebook `02_Modeling.ipynb`. Bàn giao cấu hình/tham số, mô hình đã chọn và cách tái lập thí nghiệm. |
| **Thành viên 3 — Đánh giá, giải thích & tích hợp** | Thiết kế đánh giá, phân tích lỗi, trực quan hóa và tổng hợp kết quả | Chốt metric phù hợp với mất cân bằng lớp; đánh giá trên test set giữ nguyên phân bố, lập bảng so sánh và confusion matrix. Phân tích False Positive/False Negative, ROC/PR, bias–variance và giới hạn; thử SHAP như nội dung mở rộng nếu khả thi. Phụ trách `src/evaluation/`, `src/visualization/`, notebook `03_Evaluation.ipynb` và tích hợp pipeline/báo cáo kết quả. Bàn giao biểu đồ, bảng số liệu, phân tích lỗi và kết luận có căn cứ. |

### Phần việc chung và phối hợp

1. Cả nhóm thống nhất câu hỏi nghiên cứu, cách diễn giải nhãn, quy tắc chia dữ liệu và metric trước khi so sánh mô hình.
2. Mỗi thành viên làm việc trên nhánh riêng, ghi lại thay đổi và mở pull request để ít nhất một thành viên khác review.
3. Không commit dữ liệu gốc, thông tin xác thực Kaggle hoặc mô hình sinh ra; chỉ commit mã nguồn, cấu hình, notebook và tài liệu cần thiết.
4. Mỗi thành viên rà soát phần mình phụ trách; thành viên 3 phối hợp kiểm tra kết quả đầu-cuối, nhưng toàn nhóm cùng chịu trách nhiệm về tính đúng đắn và nội dung báo cáo.
5. Trước khi nộp, cả nhóm kiểm tra notebook chạy được theo hướng dẫn README, thống nhất phiên bản kết quả, ghi rõ hạn chế và chuẩn bị phần trình bày.
