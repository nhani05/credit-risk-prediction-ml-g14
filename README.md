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

## Phân công công việc (nhóm 3 thành viên, hoàn thành trong 5 tuần)

> Thay “Thành viên 1/2/3” bằng tên thực tế. Đây là phân công theo giai đoạn và trách nhiệm chính, không phải chia cứng theo tầng. Mỗi người đều tham gia ít nhất một phần của pipeline trên cả hai bộ dữ liệu và phải hiểu luồng từ dữ liệu đến đánh giá.

| Thời gian | Thành viên 1 | Thành viên 2 | Thành viên 3 |
|---|---|---|---|
| **Tuần 1 — Dữ liệu và khung pipeline** | EDA, kiểm tra chất lượng, nhãn và giới hạn của German Credit; đề xuất tiền xử lý. | EDA, kiểm tra chất lượng, nhãn và giới hạn của Give Me Some Credit; đề xuất tiền xử lý. | Dựng/thống nhất giao diện pipeline và cấu hình dùng chung; tích hợp thử trên cả hai bộ, bổ sung test kiểm tra chia train/test và chống leakage. |
| **Tuần 2 — Tiền xử lý và baseline** | Hoàn thiện tiền xử lý German Credit; chạy baseline dự đoán lớp đa số và Logistic Regression trên cả hai bộ. | Hoàn thiện tiền xử lý Give Me Some Credit; chạy Decision Tree, KNN và SVM trên cả hai bộ. | Đóng gói tiền xử lý theo pipeline, kiểm tra leakage; chạy Random Forest/Gradient Boosting phù hợp với dependency hiện có trên cả hai bộ. |
| **Tuần 3 — So sánh và chốt mô hình** | Kiểm tra, giải thích kết quả baseline; phân tích ban đầu FP/FN và các vấn đề dữ liệu. | Chạy ensemble/boosting đã thống nhất, lưu cấu hình và kết quả có thể tái lập. | Thiết kế cross-validation và tuning chỉ trên train; tổng hợp bảng so sánh. Cả nhóm cùng chọn mô hình/quy trình trước khi mở test để đánh giá cuối. |
| **Tuần 4 — Đánh giá và viết báo cáo** | Phân tích lỗi, confusion matrix, ROC/PR và fairness phù hợp với dữ liệu; viết phần dữ liệu, EDA và giới hạn. | Đánh giá calibration, thử cost-sensitive learning; giải thích chi phí FP/FN và viết phần phương pháp/mô hình. | Thực hiện SHAP cho mô hình đã chọn, tạo bảng/biểu đồ đánh giá và viết phần kết quả. SHAP không được diễn giải như bằng chứng nhân quả hay công bằng. |
| **Tuần 5 — Hoàn thiện, in và trình bày** | Review chéo kết quả và kết luận; kiểm tra khớp giữa nội dung báo cáo với thí nghiệm đã chạy. | Tổng biên tập bản in: định dạng, mục lục, đánh số hình/bảng, in thử và phối hợp in/đóng bìa. Bắt đầu quy trình in trước hạn nộp ít nhất 3–4 ngày. | Tích hợp báo cáo, rà soát tính tái lập và hoàn thiện phụ lục kỹ thuật; cả nhóm cùng xác nhận nội dung cuối và chuẩn bị thuyết trình. |

### Nguyên tắc phối hợp và cân bằng khối lượng

1. **Đầu ra rõ ràng:** mỗi đầu việc phải có kết quả bàn giao (mã/notebook, test, bảng số liệu, đoạn báo cáo hoặc checklist). Thành viên 3 không mặc định gánh toàn bộ tích hợp: tuần 5, thành viên 1 review kết quả, thành viên 2 chịu trách nhiệm bản in, thành viên 3 tích hợp kỹ thuật; cả nhóm cùng xác nhận.
2. **Theo dõi thời gian:** họp ngắn đầu tuần để chốt đầu việc và ước tính giờ; cuối tuần ghi nhận giờ thực tế, việc tồn và điều chỉnh. Nếu một người đang vượt đáng kể phần còn lại, chuyển việc tương đương ngay trong tuần kế tiếp, không cộng thêm việc mở rộng.
3. **Review chéo luân phiên:** Thành viên 1 review Thành viên 2; Thành viên 2 review Thành viên 3; Thành viên 3 review Thành viên 1. Review qua pull request, tập trung vào khả năng chạy lại, tính đúng đắn và chống leakage.
4. **Học toàn bộ pipeline:** cuối mỗi tuần, người phụ trách trình bày phần đã làm và cách nối với các bước còn lại. Trong tuần 5, cả ba cùng chạy lại ít nhất một pipeline hoàn chỉnh trên mỗi bộ dữ liệu và mỗi người giải thích được: chia dữ liệu → tiền xử lý → huấn luyện → chọn mô hình → đánh giá.
5. **Chống leakage:** chia train/test trước khi fit; imputer, encoder, scaler, chọn đặc trưng, ngưỡng và resampling chỉ fit trên train hoặc fold train. Giữ nguyên test cho đánh giá cuối; không dùng test để tuning hay chọn mô hình.
6. **Giới hạn phạm vi:** không gộp hai bộ dữ liệu. Các phần mở rộng ngoài kế hoạch (ví dụ SHAP nâng cao hoặc mô hình mới) chỉ làm sau khi các đầu ra cốt lõi hoàn tất và phải thay thế một việc tương đương, không làm tăng khối lượng hay ảnh hưởng hạn nộp.
7. Không commit dữ liệu, credentials hoặc mô hình sinh ra. Trước khi nộp, chạy các kiểm tra phù hợp, xác nhận hướng dẫn README và ghi rõ kiểm tra nào chưa chạy được cùng lý do.

### Nhịp làm việc trong 5 tuần

- **Mỗi tuần:** một buổi thống nhất kế hoạch và một buổi review tiến độ/đầu ra; xử lý vướng mắc sớm để không dồn việc sang tuần cuối.
- **Cuối tuần 2:** xác nhận hai bộ dữ liệu đều chạy được qua tiền xử lý và baseline.
- **Cuối tuần 3:** chốt mô hình và cấu hình mà không dựa vào test set.
- **Tuần 4:** hoàn tất kết quả và phần lớn bản thảo; dành tuần 5 cho rà soát, định dạng, in và thuyết trình, không mở thêm thí nghiệm lớn.
- **Tuần 5:** bàn giao nội dung cho in trước hạn nộp 3–4 ngày để có thời gian in thử và sửa lỗi.

**Phạm vi báo cáo:** ưu tiên hoàn thành pipeline tái lập, baseline, so sánh mô hình, metric phù hợp, confusion matrix và phân tích lỗi cho từng bộ dữ liệu. Calibration, cost-sensitive learning, fairness và SHAP là các nội dung mở rộng có phạm vi giới hạn; nếu thiếu thời gian, nhóm cần nêu rõ phần nào đã thực hiện và phần nào chưa, không trình bày như kết quả đã có.

#### Lưu ý

Các kết quả của German Credit và Give Me Some Credit phải được báo cáo riêng vì khác định nghĩa nhãn, đặc trưng và quần thể. Không diễn giải mức điểm giữa hai bộ như phép so sánh trực tiếp cùng một bài toán.
