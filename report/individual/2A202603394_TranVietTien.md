# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603394
- **Họ và tên:** Trần Việt Tiến
- **SEED:** 316

## 1. Bằng chứng

| Tệp yêu cầu | Trạng thái hiện tại | Ghi chú |
| --- | --- | --- |
| `day01_lab.ipynb` | Có notebook, nhưng sai tên | [`day01_spam_ml_dl_embedding_llm.ipynb`](day01_spam_ml_dl_embedding_llm.ipynb) có output, 10/15 kiểm tra tự động đạt và đủ 10 dự đoán; cần đổi tên thành `day01_lab.ipynb` trước khi nộp. |
| `lab1_report.json` | Chưa có trong repo | Ô Báo cáo cuối có log “đã lưu”, nhưng tệp JSON không có trong thư mục nộp. |
| `hand_calc.jpg` | Chưa đặt đúng tên/vị trí | Có ba ảnh tính tay ở thư mục gốc, trong đó có [`hand_calc.jpeg`](../../hand_calc.jpeg); cần chọn ảnh đầy đủ và đổi tên/di chuyển trước khi nộp. |

Notebook dùng 5.169 SMS sau khi loại 403 dòng trùng; tập train/validation/test lần lượt là 3.619/775/775 tin, trong đó test có 101 spam. Các số dưới đây được chép từ output đang lưu trong notebook, vì vậy tốc độ phụ thuộc runtime Colab đã chạy.

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression | 0,954 | 0,974 | 0,070 | 0 |
| BoW + MLP | 0,969 | 0,974 | 0,379 | 0 |
| MiniLM + kNN | 0,907 | 0,927 | 141,209 | 0 |
| MiniLM + Logistic Regression | 0,950 | 0,947 | 32,932 | 0 |
| MiniLM + MLP head | **0,975** | 0,974 | **21,296** | 0 |
| LLM nhỏ zero-shot | — | 0,976 | 2.240,310 | 0,00000* |
| LLM lớn zero-shot | — | 1,000 | 2.594,530 | 0,00000* |
| LLM nhỏ few-shot | — | 1,000 | 2.260,460 | 0,00000* |

`*` Cột chi phí LLM bằng 0 vì router đang dùng không trả bảng giá; không có nghĩa là LLM miễn phí. LLM chỉ được chấm trên SUB 40 nên không so trực tiếp F1 test của chúng với các mô hình còn lại. Trên SUB nhỏ, chênh lệch dưới khoảng 0,05 F1 không đủ để xếp hạng chắc chắn.

MLP trên embedding là phương án local có F1 test cao nhất (0,975) và vẫn đạt SLA 50 ms. kNN dùng cùng encoder nhưng vượt SLA rõ rệt, cho thấy bước tìm láng giềng/embedding không tự động phù hợp với inference thời gian thực.

## 3. Dự đoán và thực tế

1. **F1 của BoW + LogReg.** Tôi dự đoán F1 khoảng 0,85 và precision cao hơn recall. Kết quả thực là F1 test 0,954, precision 0,989 và recall 0,921. Tôi đã đánh giá thấp dữ liệu test thông thường: nhiều mẫu vẫn có các từ khoá rất mạnh như `call`, `txt`, `free`, nên mô hình BoW tách được tốt. Recall thấp hơn precision đúng như dự đoán; các spam diễn đạt lạ vẫn bị lọt.
2. **MLP và ReLU.** Tôi dự đoán MLP chỉ bằng hoặc kém nhẹ vì số tham số lớn dễ overfit. Trên test, MLP đạt F1 0,969, cao hơn LogReg 0,015; bỏ ReLU vẫn cho F1 0,969 trong lần chạy này. Vì hai lớp tuyến tính liên tiếp vẫn tương đương một phép biến đổi tuyến tính, ReLU không tạo lợi thế rõ ràng với BoW và tập dữ liệu nhỏ; bản MLP vẫn có thể cải thiện nhờ cách tối ưu/early stopping chứ không phải vì “deep” luôn tốt hơn.
3. **Độ bền ở bộ 50 case khó.** Tôi dự đoán BoW đạt 70–80%, embedding 80–90% và LLM 90–100%. Kết quả F1 case khó: BoW + LogReg 0,378; BoW + MLP 0,410; MiniLM + MLP head 0,667; LLM zero-shot 0,967. Sai lệch lớn nhất là đã quá lạc quan về BoW: F1 SUB 0,974 không phản ánh năng lực trước spam tự nhiên, thao túng và né từ khoá.

## 4. Phân tích 50 case khó

BoW + LogReg tụt mạnh nhất: từ F1 SUB 0,974 xuống 0,378, giảm 0,596. Test thông thường vẫn có mẫu gần train hoặc có token quen thuộc; 50 case khó chủ đích đổi cách viết và ngữ cảnh nên đánh giá test thường trở nên lạc quan.

- **Case #19 — `natural_spam`:** tin nhắn có giọng thân mật, hứa “một điều đặc biệt” và yêu cầu trả lời để nhận liên kết. Nó thiếu các token quảng cáo quen thuộc, nên BoW dễ không kích hoạt các trọng số spam. Embedding/LLM có thêm tín hiệu từ ý nghĩa “dụ trả lời để lấy link”, nên phù hợp hơn.
- **Case #9 — `ham_looks_spam`:** “Txt me … free tickets … concert” là ham nhưng chứa `txt`, `free`, `tickets`, các token thường gắn với spam. Đây là rủi ro false positive của bộ lọc theo từ khoá. Ngay LLM phòng thủ cũng sai 1/50 case này, nên không nên xem LLM là lớp quyết định duy nhất.

Nhóm khó với BoW là spam tự nhiên, thao túng và né từ khoá: token bị biến dạng như leetspeak hoặc không dùng “free/win” làm đặc trưng bị mất. Với LLM, prompt injection là một bề mặt tấn công cần phòng thủ; trong lần chạy này, prompt phòng thủ đạt 100% ở nhóm injection và nâng F1 tổng từ 0,967 lên 0,983, nhưng vẫn cần chuẩn hoá đầu vào và lớp kiểm tra độc lập ngoài prompt.

## 5. Thử thách cá nhân

SEED 316 chọn đề #5: tự viết 4 SMS, gồm 2 spam viết leetspeak và 2 ham có `free`/`win`, sau đó so sánh BoW với MiniLM + LogReg.

Tôi đã thêm các câu `C1a1m y0ur fr33...`, `W1n a pr1ze...`, “Free pizza…” và “We win the match…”. Thiết kế này kiểm tra đúng hai giới hạn: BoW có thể bỏ sót spam khi token bị thay ký tự, đồng thời có thể chặn nhầm ham chỉ vì có `free` hoặc `win`; embedding kỳ vọng dùng được ngữ cảnh/ngữ nghĩa tốt hơn. Ô code thử thách chưa có output được lưu trong notebook, vì vậy chưa thể báo số dự đoán cụ thể một cách trung thực. Cần chạy lại ô này và chép bốn dòng kết quả trước khi nộp.

## 6. Phương án cho sản phẩm thật

Yêu cầu 10 triệu SMS/ngày tương đương khoảng 116 tin/giây trung bình và p95 dưới 50 ms/tin. Các phương án đạt p95 là BoW + LogReg, BoW + MLP, MiniLM + LogReg và MiniLM + MLP head. Tôi chọn **MiniLM + MLP head** làm mô hình chính vì đạt F1 test 0,975, p95 21,296 ms và thông lượng 120,742 tin/giây trong lần đo; đây là F1 tốt nhất trong nhóm đạt SLA.

Đánh đổi là mô hình encoder có thời gian chuẩn bị 38,757 giây và thông lượng chỉ nhỉnh hơn mức trung bình 116 tin/giây, nên triển khai thật phải đo tải đỉnh, có hàng đợi và/hoặc thêm replica thay vì kết luận một instance là đủ. BoW + LogReg là lớp rẻ, rất nhanh (p95 0,070 ms) để lọc tin chắc chắn. Một router có thể chuyển các tin lưỡng lự sang MiniLM + MLP head, còn LLM chỉ dùng cho điều tra ngoại lệ/offline.

LLM không đáp ứng SLA trực tiếp vì p95 khoảng 2,24–2,59 giây. Theo giá bài tính tay cho gpt-6-luna (100 input token và 1 output token mỗi SMS, $0,10/$0,50 mỗi một triệu token), chi phí là `(10.000.000 × 100 / 1.000.000) × 0,10 + (10.000.000 × 1 / 1.000.000) × 0,50 = $10,50/ngày`, tức khoảng **$315/tháng** (30 ngày). Con số này là giá giả định của bài, không phải chi phí router đã đo.

## 7. AI log và điều đã học

| Phần | Tôi hỏi AI gì | AI trả lời/viết gì | Tôi đã sửa hoặc kiểm chứng |
| --- | --- | --- | --- |
| Phần 4–5 | Hỗ trợ cấu hình router, rà soát TODO và tóm tắt kết quả | Gợi ý code gọi LLM và giải thích các chỉ số | Chạy mock, đối chiếu bảng output và sửa theo output notebook |
| Báo cáo | Tổng hợp số liệu đã lưu, kiểm tra tính nhất quán của chi phí | Gợi ý cấu trúc và phép tính | Giữ số liệu có trong notebook; sửa chi phí mẫu từ $105/ngày thành $10,50/ngày theo công thức token |

Điều quan trọng tôi hiểu thêm là F1 cao trên test thông thường chưa chứng minh mô hình bền: BoW đạt 0,954 trên test nhưng chỉ 0,378 trên bộ case khó. Embedding không tự phân loại spam; nó tạo vector nghĩa để một head nhỏ học ranh giới tốt hơn. Cuối cùng, LLM có thể tốt khi chưa có dữ liệu nhãn, nhưng độ trễ, chi phí và prompt injection khiến nó không phù hợp làm lớp lọc SMS trực tiếp ở quy mô này.

> Trước khi nộp, cần sao chép notebook đã chạy, `lab1_report.json` và một ảnh tính tay rõ ràng vào `report/individual/` với đúng tên quy định; đồng thời chạy lại ô thử thách để lưu kết quả bốn case.
