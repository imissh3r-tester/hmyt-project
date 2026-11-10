# Bài kiểm tra lần 2 — N33 — <topic_slug>

- Fork: ⚠️ ĐIỀN <URL fork hmyt-project> · Nhánh: gd5-proposed-model
- Thành viên (MSSV) và vai trò:
  - 202414830 — Pipeline tiền xử lý (ColumnTransformer, khử trùng lặp, assertion, SMOTE/SMOTETomek), viết phần EDA và Preprocessing/Data leakage
  - 202414846 — EDA và trực quan hóa, ma trận y văn, định dạng báo cáo

> Quy ước: ⚠️ ĐIỀN = nhóm phải tự điền hoặc xác minh trước khi nộp. Không có số liệu nào bên dưới được tự suy ra ngoài báo cáo IEEE và slide Tuần 7 của nhóm.

---

## A. Literature Review và research gaps

### A1. Chiến lược tìm kiếm
| Nguồn | Từ khóa / truy vấn | Khoảng năm | Số kết quả | Số bài giữ lại |
|---|---|---|---|---|
| PubMed | ⚠️ ĐIỀN (vd: breast cancer AND survival AND machine learning AND SEER) | 2021–2026 | ⚠️ ĐIỀN | ⚠️ ĐIỀN |
| Scopus | ⚠️ ĐIỀN | 2021–2026 | ⚠️ ĐIỀN | ⚠️ ĐIỀN |
| Google Scholar | ⚠️ ĐIỀN | 2021–2026 | ⚠️ ĐIỀN | ⚠️ ĐIỀN |
| Tổng sau loại trùng | | | | 20 bài trong ma trận báo cáo (A2 dưới đây chọn 14 bài liên quan nhất) |

Tiêu chí chọn / loại bài:
- Chọn: nghiên cứu dự đoán sống còn hoặc kết cục ung thư vú bằng ML/survival model, ưu tiên dùng SEER hoặc registry/bệnh viện có mô tả rõ cách chia dữ liệu; xuất bản 2021–2026.
- Loại khỏi A2 (vẫn có trong báo cáo): Soleimani et al. 2025 (dữ liệu ung thư đại trực tràng, chỉ dùng để tham khảo phương pháp); Siegel et al. 2026 (báo cáo dịch tễ, không phải mô hình dự đoán); Zhang et al. 2025 (phân tích di căn, Kaplan–Meier/Cox, không phải mô hình dự đoán); Akter/Islam 2024 (dữ liệu Bangladesh + Wisconsin chẩn đoán, không phải tiên lượng; ⚠️ tên tác giả trong bảng và danh mục tài liệu tham khảo không khớp — kiểm tra lại).

### A2. Ma trận y văn
Số trong ngoặc vuông [n] là số trong danh mục tài liệu tham khảo của báo cáo IEEE.

| # | Bài (tác giả, năm) | DOI / link | Dataset (n, nguồn) | Phương pháp | Kết quả chính (metric, giá trị) | Kiểm định ngoài? | Hạn chế chính |
|---|---|---|---|---|---|---|---|
| 1 | Moncada-Torres et al., 2021 [1] | 10.1038/s41598-021-86327-7 | Netherlands Cancer Registry, n = 36,658 | CPH, RSF, Survival SVM, XGBoost; chia ở mức bệnh nhân, 10-fold CV | XGBoost C-index ≈ 0.73, CPH ≈ 0.72 | Không | Dữ liệu một quốc gia, hồi cứu; dữ liệu thô không chia sẻ được |
| 2 | Sedighi-Maman & Mondello, 2021 [2] | 10.1016/j.ijmedinf.2021.104438 (⚠️ ma trận cũ ghi sai DOI `artmed.2021.102182`; đã sửa theo danh mục tham khảo và trang Elsevier/Int J Med Inform) | SEER 2004–2016 | Hai giai đoạn: phân loại (GLM-LASSO, XGB, MLP + SMOTE/ADASYN) rồi hồi quy số tháng sống | Giai đoạn I (1 năm): GLM-LASSO-SMOTE AUC 0.898, Acc 0.821; giai đoạn II: MAE 7.960 tháng | Không | SEER thiếu biến lâm sàng chi tiết; MLP kém GLM ở giai đoạn hồi quy |
| 3 | Manikandan, Durga & Ponnuraja, 2023 [4] | 10.1038/s41598-023-32029-1 | SEER Breast, n = 712,319 (⚠️ kiểm tra lại số n trong bài gốc) | Chọn đặc trưng (Variance Threshold + PCA) + AdaBoost, XGB, GB, NB, DT; chia theo bản ghi 70:30 | DT: Acc, Precision, Recall, F1 đều 0.98 | Không | Nhiều dữ liệu thiếu; nhóm nghi ngờ rò rỉ do PCA trước khi chia (giả thuyết, chưa chứng minh) |
| 4 | Gurcan & Soylu, 2024 [5] | 10.3390/cancers16193417 | 5 bộ ung thư; SEER Breast n = 4,024 | 10 bộ phân loại × 19 thuật toán resampling; 5-fold CV phân tầng | SEER Breast, SMOTEENN: Acc 92.87%, F1 92.88%, ROC-AUC 97.71% | Không | Chỉ tập trung resampling; không có XAI. Chưa rõ resampling áp dụng trước hay sau khi chia |
| 5 | Rajpoot et al., 2024 [7] | 10.1080/08839514.2024.2340386 | UCI Breast Cancer, ⚠️ ĐIỀN n | RF, DT, SVM, ANN + chọn đặc trưng; 10-fold CV theo bản ghi | Mô hình đề xuất: Sens 85.5%, Spec 79.3%, Precision 87% | Không | Dataset benchmark UCI, không có cohort lâm sàng thực; không XAI |
| 6 | Manir & Deshpande, 2024 [8] | 10.3390/diagnostics14100984 | BCSC n = 527,712; WBCD n = 569; SEER Breast n = 4,024 | Extra Trees, RF, XGB, LR, DT, KNN + PCA + SMOTE/Tomek; 70/30 + 10-fold CV | BCSC Extra Trees Acc 87.54%; WBCD KNN+PCA Acc 95.29%, AUC 0.9796 | Không | Mất cân bằng nặng ở BCSC; PCA làm giảm khả năng diễn giải |
| 7 | Alshamrani, 2025 [9] | 10.3390/electronics14163173 | Gamma tổng hợp n = 1,000; SEER n = 4,024 | Box–Cox + LR, SVM, RF, XGB, voting, stacking; 80/20 + SMOTE trên train | SEER: Stacking + Box-Cox + SMOTE Acc 94.53% | Không | Khó chọn λ; chi phí tính toán; không XAI |
| 8 | Baidoo & Rodrigo, 2025 [10] | 10.1371/journal.pone.0318167 | SEER Breast n = 4,024 (cùng dataset với nhóm) | CPH, RSF, DeepSurv so với RF, XGB, SVM, LightGBM + SMOTE | RSF C-index 0.72 ± 0.0117, IBS 0.08; RF AUC 0.74 ± 0.0021, Acc 0.82; XGB AUC 0.69 | Không | Mô hình ML phân loại không xử lý censoring |
| 9 | Noman et al., 2025 [12] | 10.1038/s41598-025-87622-3 | METABRIC, MSKCC, Duke, SEER (n = 272,252) + Baheya (Ai Cập) | RFS + LightGBM, XGB, RF dự đoán tái phát/di căn xa | C-index 0.837; LightGBM tái phát AUC 92% | Có (Baheya) | Sai số phân loại ≈ 13.55% (di căn xa), 10.53% (tái phát tại chỗ) |
| 10 | Raz et al., 2026 [14] | 10.1186/s12885-026-16283-w | Bệnh viện Hazrat Rasool, n = 3,476, 43 biến | XGB, LR, RF, DT, AdaBoost, MLP + SMOTE; nested 5-fold CV | XGB/DT/AdaBoost AUC 0.97, Acc > 90% | Không | Một bệnh viện; phải impute nhiều dữ liệu thiếu |
| 11 | Takele & Chen, 2026 [16] | 10.1038/s41598-026-40565-9 | Cohort Ethiopia, n = 1,164 | CPH, RSF, DeepSurv, SVM, RF, XGB, LGBM + Optuna | RSF C-index 0.754, IBS 0.091; RF AUC 0.73 | Không | Cỡ mẫu nhỏ; thiếu biomarker phân tử |
| 12 | Zhang et al., 2026 (TNBC) [17] | 10.21037/tcr-2025-aw-2462 | SEER 2010–2015, n = 27,256 TNBC | Cox nomogram so với RSF; chia 7:3 | RSF CSS AUC test 0.845 (1 năm), 0.815 (3 năm), 0.785 (5 năm); OS C-index 0.742 | Không | SEER thiếu liều và phác đồ hóa trị |
| 13 | Liu et al., 2026 [19] | 10.1186/s12885-026-16166-0 | SEER 2010–2020, n = 41,124 + ngoài GMUCH Trung Quốc n = 671 | RSF, Survival-GBM, Survival-XGB, LASSO-Cox dự đoán BCSS 3/5/7/10 năm | RSF tốt nhất; SHAP: Stage, M stage, N stage là các biến quan trọng nhất | Có (GMUCH, n = 671) | Thiếu Ki-67, hóa trị; overfit nhẹ ở mô hình cây |
| 14 | Shen et al., 2026 [20] | 10.3389/fonc.2026.1830769 | Bệnh viện Maoming, n = 1,056 (2012–2021) | Cox, LASSO, Elastic-Net, DT, RSF, XGB, GBM; 70/30 + 5-fold CV | XGB AUC validation 0.779 (1 năm), 0.751 (3 năm), 0.804 (5 năm) | Không | Một trung tâm; tập validation nhỏ (n = 316) |

⚠️ ĐIỀN: mỗi thành viên mở lại từng bài mình phụ trách, đối chiếu DOI và con số với bài gốc. Các bài năm 2026 [14][16][17][18][19][20] chưa được xác minh độc lập khi soạn nháp này.

### A3. Research gaps
| Mã | Khoảng trống | Bằng chứng (# trong A2) | Dự án của nhóm giải quyết bằng cách nào |
|---|---|---|---|
| G1 | Kết quả rất cao đi kèm chia theo bản ghi hoặc resampling/PCA trước khi chia; kết quả chia theo bệnh nhân thường chỉ đạt AUC/C-index 0.72–0.75. Lưu ý: chưa bài nào được chứng minh là có rò rỉ, đây là nghi ngờ dựa trên cách mô tả phương pháp | #3 (Acc 0.98, chia theo bản ghi), #4 (AUC 0.977), #7 (Acc 94.53%) so với #1 (0.73), #8 (0.72–0.74), #11 (0.729–0.754) | Pipeline fit-on-train (ColumnTransformer), SMOTE/SMOTETomek chỉ trên train, bỏ Survival Months, khử trùng lặp, có assertion kiểm tra. Hạn chế còn lại: SEER không có patient_id nên không chia theo bệnh nhân thật được |
| G2 | Hiệu chuẩn xác suất (calibration slope, CITL, Brier) hầu như không được báo cáo trong cột kết quả chính | Không có bài nào trong #1–#14 nêu slope/CITL/Brier; chỉ #8 và #11 nêu IBS (chỉ số cho survival, không phải calibration) (⚠️ đối chiếu lại bài gốc) | Báo cáo AUC, PR-AUC, Brier, slope, CITL trên test n = 805 (slide 18). Chưa có 95% CI và đường cong calibration |
| G3 | Các bài đạt metric cao nhất không diễn giải mô hình (không XAI). Lưu ý: nhiều bài mới đã dùng SHAP nên gap này hẹp hơn báo cáo IEEE viết | Không XAI: #4, #5, #7; có SHAP: #1, #8, #11, #12, #13, #14 | Hồi quy logistic làm mốc, báo cáo OR kèm 95% CI (slide 15: Grade OR 1.75, ER dương OR 0.49) |
| G4 | Thiếu kiểm định ngoài; phần lớn dùng một registry hoặc một bệnh viện | Chỉ #9 và #13 có kiểm định ngoài; #1–#8, #10–#12, #14 không | Dự án không giải quyết được (chỉ có tập SEER nội bộ). Ghi nhận là giới hạn, hướng phát triển |

### A4. Thẩm định 1 bài quan trọng nhất bằng PROBAST+AI
Bài số #: 8 (Baidoo & Rodrigo, 2025 [10]) · Lý do chọn: dùng đúng dataset SEER n = 4,024 như nhóm, có cả mô hình survival và ML phân loại nên so sánh trực tiếp được.

⚠️ ĐIỀN: bảng dưới là nháp dựa trên mô tả trong ma trận; nhóm đọc lại bài gốc và điền số trang trước khi nộp.

| Miền | Nguy cơ sai lệch | Căn cứ (trích dẫn hoặc số trang trong bài) |
|---|---|---|
| Người tham gia và nguồn dữ liệu | Không rõ | Bản SEER công khai n = 4,024 đã qua lọc (tiêu chí chọn/loại chưa rõ); chỉ đánh giá nội bộ. ⚠️ ĐIỀN trang |
| Biến dự báo | Thấp (dự kiến) | Biến lâm sàng thường quy. ⚠️ ĐIỀN: xác nhận không dùng biến sau T0 |
| Kết cục | Cao với nhánh ML phân loại | Ma trận ghi: mô hình ML chuẩn không xử lý censoring và thời gian sống. Nhánh survival (RSF, DeepSurv) ít rủi ro hơn. ⚠️ ĐIỀN trang |
| Phân tích | Không rõ, nghiêng về cao | Dùng SMOTE + CV: ⚠️ ĐIỀN xác nhận SMOTE chỉ áp trong từng fold train; chưa thấy báo cáo calibration |
Tính áp dụng cho bài toán của nhóm: Cùng dataset và cùng quần thể (SEER, n = 4,024) nên áp dụng tốt. C-index/IBS (survival) không so sánh trực tiếp được với AUC phân loại nhị phân của nhóm.

---

## B. Kết quả và timeline

### B1. Kết quả hiện có
Tập đánh giá: chia 80/20 phân tầng theo Status, seed cố định, sau khi bỏ 1 dòng trùng (N = 4,023; train 3,218, test 805). Tập test chứa ≈ 123 ca Dead (15.3% × 805).

| Mô hình | Tập đánh giá | Metric chính | Metric phụ | Nguồn số liệu |
|---|---|---|---|---|
| Baseline 1 (Mốc 2): Logistic Regression | Test, hold-out 20%, n = 805 | ROC-AUC 0.7219; PR-AUC 0.3574. 95% CI: ⚠️ ĐIỀN (chưa có; ước lượng thô theo Hanley–McNeil ≈ ±0.05) | Ngưỡng Youden J* = 0.528; Recall (Dead) 56.9%; Spec 71.8%; Brier 0.1161; calibration slope 0.865; CITL 0.0003. CV 5-fold: ROC-AUC 0.7400 ± 0.0159, PR-AUC 0.3963, Recall 63.9% | Slide Tuần 7 (slide 16, 18); ⚠️ ĐIỀN đường dẫn notebook/CSV/W&B |
| Baseline khác (Mốc 2) | Test, n = 805 | RF: ROC-AUC 0.7200, PR-AUC 0.3360; XGBoost (depth 3): 0.7224, 0.3290; Extra Trees: 0.7288, 0.3571 | Brier 0.1162 / 0.1162 / 0.1167 | Slide 18; ⚠️ ĐIỀN đường dẫn |
| Mô hình đề xuất (Mốc 3) | ⚠️ ĐIỀN | ⚠️ ĐIỀN (chưa có trong báo cáo và slide) | ⚠️ ĐIỀN | ⚠️ ĐIỀN |

Nhận xét tự kiểm tra:
- Brier của hồi quy logistic trên test (0.1161) chỉ tốt hơn một chút so với mô hình "luôn dự đoán tỷ lệ nền" (0.153 × 0.847 ≈ 0.130).
- Brier CV của LR là 0.1991 (slide 16) cao hơn nhiều Brier test 0.1161 và cao hơn mức nền 0.130, trong khi CITL test gần bằng 0 (0.0003). ⚠️ ĐIỀN: xác nhận xác suất trên test có được hiệu chỉnh lại không, và hiệu chỉnh/chọn ngưỡng J* trên tập nào (phải là train hoặc validation, không phải test).

### B2. Ablation (Mốc 3)
⚠️ ĐIỀN: chưa có số liệu trong tài liệu. Gợi ý các cấu hình có thể chạy ngay với pipeline hiện có:

| Cấu hình | Metric chính (ROC-AUC / PR-AUC) | Chênh lệch so với mô hình đầy đủ | Nguồn |
|---|---|---|---|
| Mô hình đầy đủ | ⚠️ ĐIỀN | 0 | ⚠️ ĐIỀN |
| Bỏ SMOTETomek (dùng class_weight) | ⚠️ ĐIỀN | ⚠️ ĐIỀN | ⚠️ ĐIỀN |
| Đổi SMOTETomek thành SMOTE | ⚠️ ĐIỀN | ⚠️ ĐIỀN | ⚠️ ĐIỀN |
| Bỏ Positive_Node_Ratio | ⚠️ ĐIỀN | ⚠️ ĐIỀN | ⚠️ ĐIỀN |
| Bỏ Yeo-Johnson (chỉ chuẩn hóa) | ⚠️ ĐIỀN | ⚠️ ĐIỀN | ⚠️ ĐIỀN |
| Bỏ 6th Stage (VIF 12.79) | ⚠️ ĐIỀN | ⚠️ ĐIỀN | ⚠️ ĐIỀN |

### B3. Đối chiếu với y văn
- So với #8 Baidoo & Rodrigo (cùng SEER n = 4,024): RF AUC 0.74 ± 0.0021; CV AUC của nhóm 0.74 (RF 0.7401, LR 0.7400) và test AUC 0.72–0.73. Kết quả cùng mức. Điểm không công bằng: bài kia có thể dùng cách chia khác (train/validation/test + CV), mức sống còn được xử lý khác (có mô hình survival), và Acc 0.82 không so được với Recall/PR-AUC của nhóm.
- So với #1 Moncada-Torres (C-index ≈ 0.73): khác dataset (NCR n = 36,658 so với 4,023), khác metric (C-index theo thời gian so với AUC nhị phân cuối theo dõi), khác bài toán (survival so với phân loại).
- So với #4 Gurcan & Soylu (ROC-AUC 97.71% trên cùng SEER 4,024) và #7 Alshamrani (Acc 94.53%): cách chia, thứ tự resampling và metric khác. Chênh lệch lớn (0.72 so với 0.98) nên nhóm không coi là so sánh công bằng và chưa kết luận nguyên nhân (có thể do resampling hoặc chia trước/sau, nhưng cần đọc lại bài gốc). Accuracy cũng không nên so với Recall/PR-AUC của nhóm do lớp lệch 84.7/15.3.

### B4. Timeline đến khi hoàn thành
⚠️ ĐIỀN: bản đề xuất, nhóm sửa lại người phụ trách và ngày thực tế. Ngày tính từ Tuần 10 = 09/11/2026.

| Việc | Phụ trách | Mốc | Ngày dự kiến | Trạng thái |
|---|---|---|---|---|
| EDA, tiền xử lý, chống leakage, 7 mô hình mốc, CV | 202414830, 202414846 | M2–M3 (tuần 10) | 09/11/2026 | Xong (slide Tuần 7; ⚠️ ĐIỀN link notebook) |
| Mô hình đề xuất + ablation (B2) | ⚠️ ĐIỀN | M3 (tuần 10) | ⚠️ ĐIỀN | ⚠️ ĐIỀN |
| Bootstrap 95% CI cho AUC, Brier, slope, CITL; đường cong calibration | 202414830 | M4 (tuần 12) | 23–27/11/2026 | Chưa |
| Xác nhận ngưỡng J* và hiệu chuẩn chỉ dùng train/validation | 202414830 | M4 (tuần 12) | 23–27/11/2026 | Chưa |
| Datasheet, checksum, giấy phép; điền checklist TRIPOD+AI | 202414846 | M4 (tuần 12) | 23–27/11/2026 | Chưa |
| Thử nhãn theo mốc thời gian cố định (vd: sống sau 5 năm) từ Survival Months | cả nhóm | M4 (tuần 12) | 23–27/11/2026 | Chưa |
| Báo cáo cuối, PR cuối cùng | cả nhóm | Final (tuần 15) | 14–18/12/2026 | Chưa |

### B5. Rủi ro / đang vướng / câu hỏi cho GV
- SEER không có patient_id: sau khi bỏ 1 dòng trùng, các dòng trùng vector đặc trưng khác nhau vẫn có thể nằm ở cả train và test (rò rỉ còn lại); số dòng trùng chéo mà notebook báo cáo là ⚠️ ĐIỀN.
- Nhãn Status (Alive/Dead) lấy ở cuối theo dõi, không có mốc thời gian cố định và không loại bệnh nhân bị censoring; bệnh nhân còn sống nhưng theo dõi ngắn bị tính như sống sót. Câu hỏi cho GV: có nên chuyển sang nhãn theo năm (vd 5 năm) không?
- Kết quả ~0.72 AUC, Brier chỉ nhỉnh hơn mức nền: mô hình phân biệt ở mức vừa phải.
- Test chỉ có ≈ 123 ca Dead nên CI rộng; chưa có CI.
- Cơ sở pháp lý dùng dữ liệu: báo cáo ghi "Open Access" trên Kaggle, nhưng dữ liệu gốc là SEER (có thỏa thuận sử dụng). Câu hỏi cho GV: bản Kaggle này có đủ căn cứ dùng cho môn học không?
- Mâu thuẫn trong tài liệu của nhóm cần thống nhất: báo cáo IEEE nói dùng SMOTE, slide dùng SMOTETomek; tuổi đỉnh 40–60 (báo cáo) so với 45–65 (slide).

---

## C. Tự đánh giá dự án theo Phụ lục FAIR, TRIPOD+AI, PROBAST+AI

### C1. FAIR — dữ liệu
| Mục | Trả lời | Bằng chứng |
|---|---|---|
| Nguồn, phiên bản, ngày tải; checksum | Breast Cancer Patient Dataset trên Kaggle, trích từ bản cập nhật SEER tháng 11/2017. n = 4,024 dòng, 16 cột. Ngày tải: ⚠️ ĐIỀN. Checksum: chưa có (⚠️ ĐIỀN: chạy `sha256sum <file>` và ghi vào đây) | Báo cáo IEEE, mục Dataset; ⚠️ ĐIỀN đường dẫn file |
| Giấy phép / DUA, căn cứ | Báo cáo ghi "Open Access" trên Kaggle. ⚠️ ĐIỀN: dán link trang Kaggle và tên giấy phép chính xác; kiểm tra điều khoản SEER Research Data Agreement | ⚠️ ĐIỀN link |
| Datasheet | Quần thể: bệnh nhân ung thư vú trong SEER (⚠️ ĐIỀN: năm chẩn đoán và tiêu chí lọc theo mô tả trên Kaggle). Biến: Age, Race, Marital Status, Tumor Size (mm), T Stage, N Stage, 6th Stage, A Stage, Grade, differentiate (trùng 100% với Grade, đã bỏ), ER/PR Status, Regional Node Examined/Positive, Survival Months (đã bỏ), Status. Thiếu: không có biến điều trị, Ki-67, HER2, patient_id. Không thiếu dữ liệu (0 null) | Báo cáo IEEE mục III; slide 6, 10 |
| Không có định danh bệnh nhân | Không có patient_id, tên, mã bệnh án; chỉ có biến nhân khẩu học thô. Không có dữ liệu bệnh nhân thật nào được đưa vào công cụ AI (theo khai báo trong báo cáo) | Báo cáo IEEE mục AI Usage; ⚠️ ĐIỀN |

### C2. Hướng dẫn báo cáo
- Câu hỏi nghiên cứu: Từ các biến lâm sàng, nhân khẩu học và bệnh học có tại thời điểm chẩn đoán, có thể dự đoán bệnh nhân ung thư vú trong SEER còn sống hay đã tử vong (Alive/Dead) tốt đến mức nào, và xác suất dự đoán có hiệu chuẩn tốt không?
- Hướng dẫn báo cáo phù hợp: TRIPOD+AI. Lý do: đây là phát triển và kiểm định nội bộ một mô hình dự đoán tiên lượng. Không dùng STARD-AI (dành cho nghiên cứu độ chính xác chẩn đoán so với chuẩn tham chiếu) và DECIDE-AI (dành cho đánh giá lâm sàng giai đoạn sớm của công cụ AI hỗ trợ quyết định, nhóm chưa triển khai trên người dùng).
- Số mục checklist đã đáp ứng / tổng số mục: ⚠️ ĐIỀN / 27 (TRIPOD+AI); các mục còn thiếu dự kiến: kích thước mẫu có căn cứ, cách xử lý/báo cáo CI, đường cong calibration, mô tả thiếu dữ liệu và công bố mã/dữ liệu. ⚠️ ĐIỀN
- Kết cục, nhãn, T0, biến có sẵn tại T0:
  - Kết cục: Status (Alive = 0, Dead = 1), tử vong do mọi nguyên nhân tính đến cuối theo dõi (không có mốc thời gian cố định).
  - T0 (đề xuất): sau phẫu thuật và có báo cáo giải phẫu bệnh (cần có Tumor Size, Grade, số hạch, ER/PR). ⚠️ ĐIỀN: xác nhận nhóm chọn T0 này.
  - Biến tại T0: Age, Race, Marital Status, Tumor Size, T/N/6th/A Stage, Grade, ER/PR Status, Regional Node Examined/Positive, Positive_Node_Ratio (đặc trưng suy ra).
  - Đã loại: Survival Months (biến thuộc kết cục), differentiate (trùng Grade).

### C3. PROBAST+AI — tự thẩm định dự án của nhóm
| Miền | Nguy cơ sai lệch | Căn cứ trong repo (file, dòng) | Việc sẽ làm để giảm nguy cơ |
|---|---|---|---|
| Người tham gia và nguồn dữ liệu | Không rõ, nghiêng về cao | Dữ liệu là bản Kaggle trích từ SEER 2017, tiêu chí lọc ban đầu không do nhóm kiểm soát; không có kiểm định ngoài. ⚠️ ĐIỀN file/dòng | Ghi rõ tiêu chí chọn/loại trong datasheet; nêu giới hạn về quần thể và thời kỳ |
| Biến dự báo | Không rõ | Báo cáo IEEE mục IV; 6th Stage kết hợp T và N (VIF 12.79) nên trùng lặp thông tin; một số biến chỉ có sau phẫu thuật. ⚠️ ĐIỀN file/dòng | Xác định T0 rõ; thử bỏ 6th Stage (ablation) |
| Kết cục (chú ý incorporation bias) | Cao | Nhãn Status không có mốc thời gian cố định và không xử lý censoring; gộp mọi nguyên nhân tử vong. Survival Months đã bỏ khỏi biến dự báo (slide 10), không có biến đã dùng để định nghĩa kết cục trong biến dự báo | Thử nhãn theo 5 năm (từ Survival Months); thảo luận bệnh nhân bị censoring |
| Phân tích | Cao (chưa xác nhận) | Không có patient_id; chỉ bỏ 1 dòng trùng (slide 10); chia 80/20 phân tầng; fit-on-train (ColumnTransformer), SMOTE/SMOTETomek chỉ trên train; chưa có 95% CI; ngưỡng J* và hiệu chuẩn có thể dùng test; test chỉ ≈ 123 ca Dead | Bootstrap CI; xác nhận J* và hiệu chuẩn chỉ dùng train/validation; tính cỡ mẫu theo Riley et al. (EPV tổng: 616 ca / ~18 biến ≈ 34, chỉ là mốc tham khảo); báo cáo đường cong calibration |
Tính áp dụng: Chỉ áp dụng cho bệnh nhân ung thư vú trong SEER giai đoạn chẩn đoán tương ứng; chưa có kiểm định ngoài nên chưa nên dùng để hỗ trợ quyết định lâm sàng thực tế. Dùng để nghiên cứu và so sánh phương pháp.

### C4. Bảng kiểm nhanh
| Mục | Đạt / Chưa | Bằng chứng |
|---|---|---|
| Datasheet | Chưa (mới có mô tả dataset trong báo cáo, chưa có datasheet riêng) | Báo cáo IEEE mục III |
| Phiên bản + checksum dữ liệu | Chưa | Mới ghi "bản cập nhật tháng 11/2017"; chưa có checksum |
| Không định danh; căn cứ dùng dữ liệu | Một phần | Không có định danh; căn cứ giấy phép chưa xác minh |
| Chọn đúng hướng dẫn báo cáo + checklist đã điền | Chưa | Đã chọn TRIPOD+AI (C2); checklist chưa điền |
| Định nghĩa kết cục, nhãn, T0 | Một phần | Có nhãn Status; chưa chốt T0 và mốc thời gian |
| Chia theo bệnh nhân; tiền xử lý fit chỉ trên train | Chưa (chia theo bệnh nhân) / Đạt (fit-on-train) | Không có patient_id; chia 80/20 theo dòng sau bỏ trùng. Fit-on-train và SMOTE chỉ trên train: báo cáo IEEE mục IV, slide 10–11 |
| Mô hình mốc hồi quy logistic | Đạt | Slide 16, 18 |
| AUC + calibration + Brier, kèm 95% CI | Một phần | Có AUC, Brier, slope, CITL (slide 18); chưa có 95% CI, chưa có đường cong calibration |
| Cỡ mẫu có đủ không; giới hạn tính áp dụng | Chưa | Chưa tính cỡ mẫu chính thức; giới hạn nêu ở C3 |

---

## D. Khai báo dùng AI
| Công cụ | Dùng cho phần nào | Nhóm đã kiểm chứng lại bằng cách nào |
|---|---|---|
| Google NotebookLM | Đọc và trích số liệu từ các bài báo để lập ma trận y văn | ⚠️ ĐIỀN (đối chiếu lại số liệu và DOI với bài gốc) |
| Anthropic Claude | Hỗ trợ viết/sửa mã vẽ biểu đồ và pipeline tiền xử lý scikit-learn; soạn nháp file kiem_tra_lan_2.md này từ báo cáo và slide của nhóm | Mã đã chạy và kiểm tra leakage (theo báo cáo). Với file này: ⚠️ ĐIỀN (nhóm đã đọc lại, sửa và đối chiếu từng con số với slide/notebook/bài gốc) |
| Google Gemini | Gợi ý cấu trúc báo cáo, chỉnh văn phong | ⚠️ ĐIỀN |
| Zotero | Quản lý trích dẫn, định dạng IEEE | Kiểm tra thủ công danh mục tham khảo (theo báo cáo); ⚠️ ĐIỀN: đã sửa DOI [2] và tác giả [6] |
