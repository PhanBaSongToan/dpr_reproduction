from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "dpr_training_negative_sampling_handbook.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))

NAVY = colors.HexColor("#12355B")
BLUE = colors.HexColor("#1D70A2")
LIGHT = colors.HexColor("#EAF3F8")
PALE = colors.HexColor("#F7FAFC")
GRAY = colors.HexColor("#536273")
GREEN = colors.HexColor("#E7F4EA")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitleX", fontName="Arial-Bold", fontSize=25, leading=30, alignment=TA_CENTER, textColor=NAVY, spaceAfter=10))
styles.add(ParagraphStyle("SubX", fontName="Arial", fontSize=11, leading=16, alignment=TA_CENTER, textColor=GRAY))
styles.add(ParagraphStyle("H1X", fontName="Arial-Bold", fontSize=18, leading=23, textColor=NAVY, spaceBefore=2, spaceAfter=8))
styles.add(ParagraphStyle("H2X", fontName="Arial-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle("BodyX", fontName="Arial", fontSize=9.2, leading=13.5, textColor=colors.HexColor("#202A33"), spaceAfter=5))
styles.add(ParagraphStyle("SmallX", fontName="Arial", fontSize=8, leading=10.5, textColor=GRAY))
styles.add(ParagraphStyle("CodeX", fontName="Courier", fontSize=8.1, leading=10.8, textColor=colors.HexColor("#14324A")))
styles.add(ParagraphStyle("CellX", fontName="Arial", fontSize=8.2, leading=10.5, textColor=colors.HexColor("#202A33")))
styles.add(ParagraphStyle("CellBoldX", fontName="Arial-Bold", fontSize=8.3, leading=10.5, textColor=NAVY))


def P(text, style="BodyX"):
    return Paragraph(text, styles[style])


def hf(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page > 1:
        canvas.setStrokeColor(BLUE)
        canvas.setLineWidth(0.6)
        canvas.line(18 * mm, h - 16 * mm, w - 18 * mm, h - 16 * mm)
        canvas.setFont("Arial-Bold", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(18 * mm, h - 12 * mm, "DPR Training and Negative Sampling Handbook")
    canvas.setFont("Arial", 8)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(w - 18 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


frame = Frame(18 * mm, 18 * mm, A4[0] - 36 * mm, A4[1] - 40 * mm, id="normal")
doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=23 * mm, bottomMargin=18 * mm)
doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=hf)])
S = []

# Cover
S += [Spacer(1, 35 * mm), P("DPR TRAINING", "TitleX"), P("Negative Sampling Handbook", "TitleX"), P("Dense Passage Retrieval for Open-Domain Question Answering", "SubX"), Spacer(1, 10 * mm), P("Tài liệu học và thuyết trình trên bảng - không phụ thuộc vào việc show code", "SubX"), Spacer(1, 28 * mm), P("Mục tiêu", "H2X"), P("Hiểu cách DPR học question và passage trong cùng một không gian vector, cách tạo negative passages và cách loss cập nhật hai encoder.", "BodyX"), Spacer(1, 30 * mm), P("Project: dpr_reproduction", "SmallX")]
S.append(PageBreak())

# 1 overview
S += [P("1. Bức tranh tổng thể", "H1X"), P("DPR là một dense retriever. Nhiệm vụ của nó không phải sinh câu trả lời, mà là xếp hạng passage có khả năng chứa câu trả lời cao hơn các passage khác.", "BodyX"), P("Luồng chính", "H2X"), P("Question q -> Question Encoder -> vector q<br/>Passage p -> Passage Encoder -> vector p<br/>vector q + vector p -> dot product -> relevance score", "CodeX"), P("Hai encoder", "H2X"), P("Question encoder biến câu hỏi thành query embedding. Passage encoder biến passage thành passage embedding. Hai encoder được fine-tune để cặp đúng có score cao và cặp sai có score thấp.", "BodyX"), P("Mục tiêu học", "H2X"), P("score(q, p+) > score(q, p-)", "CodeX"), P("Trong đó p+ là positive passage và p- là negative passage. Positive hay negative không phụ thuộc tên p1, p2; nó phụ thuộc nhãn của dataset.", "BodyX")]
S.append(PageBreak())

# 2 similarity
S += [P("2. Embedding và dot product", "H1X"), P("Sau khi encoder xử lý văn bản, mỗi question và passage trở thành một vector số có cùng số chiều D.", "BodyX"), P("Ví dụ", "H2X"), P("q = [1, 2, 3]<br/>p = [4, 5, 6]<br/><br/>q dot p = 1x4 + 2x5 + 3x6 = 32", "CodeX"), P("DPR dùng dot product vì phép tính nhanh, có thể tính thành ma trận và phù hợp với Maximum Inner Product Search. Passage embeddings có thể được tạo trước, sau đó query chỉ cần tìm các passage có inner product cao.", "BodyX"), P("Ma trận similarity", "H2X"), P("Q: [B x D]<br/>P: [B x D]<br/>S = QP^T: [B x B]", "CodeX"), P("S[i][j] là score giữa question thứ i và passage thứ j. Mục tiêu là làm score trên đường chéo cao hơn các score ngoài đường chéo.", "BodyX")]
S.append(PageBreak())

# 3 positives
S += [P("3. Positive passage và negative passage", "H1X"), P("Một training example thường có một question và ít nhất một passage đúng. Các passage sai được dùng để dạy model phân biệt.", "BodyX"), P("Positive", "H2X"), P("p+ là passage chứa answer hoặc được gán là relevant trong dữ liệu DPR.", "BodyX"), P("Negative", "H2X"), P("p- là passage không chứa answer hoặc không liên quan. Negative có thể là random, BM25 hoặc gold passage của question khác.", "BodyX"), P("Mục tiêu", "H2X"), P("Positive score increases -> good<br/>Negative score decreases -> good<br/>Positive score lower than negative score -> ranking error", "CodeX"), P("Tại sao cần negative?", "H2X"), P("Nếu chỉ cho model xem cặp question-passage đúng, model biết cặp nào đúng nhưng không học được cách loại passage sai. Negative tạo ra sự so sánh cần thiết cho metric learning.", "BodyX")]
S.append(PageBreak())

# 4 in-batch
S += [P("4. In-batch negatives", "H1X"), P("Giả sử mini-batch có 3 cặp đúng:", "BodyX"), P("q1 -> p1 positive<br/>q2 -> p2 positive<br/>q3 -> p3 positive", "CodeX"), P("Khi tính toàn bộ ma trận, positive passage của question khác trở thành negative:", "BodyX")]
mat = [[P("", "CellBoldX"), P("p1", "CellBoldX"), P("p2", "CellBoldX"), P("p3", "CellBoldX")], [P("q1", "CellBoldX"), P("POS", "CellBoldX"), P("NEG", "CellX"), P("NEG", "CellX")], [P("q2", "CellBoldX"), P("NEG", "CellX"), P("POS", "CellBoldX"), P("NEG", "CellX")], [P("q3", "CellBoldX"), P("NEG", "CellX"), P("NEG", "CellX"), P("POS", "CellBoldX")]]
mt = Table(mat, colWidths=[28 * mm, 28 * mm, 28 * mm, 28 * mm])
mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("BACKGROUND", (0, 0), (0, -1), LIGHT), ("BACKGROUND", (1, 1), (-1, -1), PALE), ("BACKGROUND", (1, 1), (1, 1), GREEN), ("BACKGROUND", (2, 2), (2, 2), GREEN), ("BACKGROUND", (3, 3), (3, 3), GREEN), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B6C9D5")), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
S += [mt, Spacer(1, 5 * mm), P("Với batch size B:", "H2X"), P("Mỗi question có 1 positive và B - 1 in-batch negatives. Toàn batch tạo B x B question-passage scores mà không cần retrieval thêm.", "BodyX"), P("Đây là lý do DPR dùng mini-batch lớn: batch càng lớn thì thường có nhiều negative hơn, nhưng chi phí bộ nhớ cũng tăng.", "BodyX")]
S.append(PageBreak())

# 5 hard negatives
S += [P("5. BM25 hard negatives", "H1X"), P("Hard negative là passage sai nhưng có nhiều từ hoặc tín hiệu lexical giống question. Nó khó hơn random negative và giúp model học phân biệt tinh tế hơn.", "BodyX"), P("Quy trình", "H2X"), P("Question -> BM25 top-k candidates -> remove positive -> choose highest-ranked remaining -> BM25 hard negative", "CodeX"), P("Ví dụ", "H2X"), P("Question: Who invented the telephone?<br/>Positive: Alexander Graham Bell invented the telephone.<br/>Hard negative: Thomas Edison worked on electrical inventions.<br/>Random negative: The weather is sunny today.", "BodyX"), P("So sánh", "H2X")]
hard = [[P("Loại negative", "CellBoldX"), P("Đặc điểm", "CellBoldX"), P("Giá trị training", "CellBoldX")], [P("Random", "CellBoldX"), P("Không liên quan", "CellX"), P("Dễ, đôi khi ít hữu ích", "CellX")], [P("In-batch", "CellBoldX"), P("Positive của question khác", "CellX"), P("Rẻ và hiệu quả", "CellX")], [P("BM25 hard", "CellBoldX"), P("Giống từ khóa nhưng sai answer", "CellX"), P("Khó, giúp học semantic distinction", "CellX")]]
ht = Table(hard, colWidths=[35 * mm, 70 * mm, 65 * mm])
ht.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B6C9D5")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
S += [ht, Spacer(1, 3 * mm), P("Luôn lọc positive passage trước khi chọn hard negative. Nếu không, có thể tạo false negative - passage đúng nhưng bị gán nhãn sai.", "BodyX")]
S.append(PageBreak())

# 6 loss
S += [P("6. Similarity matrix và loss", "H1X"), P("Với batch size B = 3, in-batch scores có dạng:", "BodyX"), P("              p1       p2       p3<br/>q1          positive  negative negative<br/>q2          negative  positive  negative<br/>q3          negative  negative  positive", "CodeX"), P("Thêm một BM25 hard negative cho mỗi question:", "H2X"), P("              p1       p2       p3       hard negative<br/>q1          positive  negative negative  hard_neg_1<br/>q2          negative  positive  negative  hard_neg_2<br/>q3          negative  negative  positive  hard_neg_3", "CodeX"), P("Kích thước logits cuối là B x (B + 1). Label vẫn là [0, 1, 2] vì positive passage nằm trên đường chéo.", "BodyX"), P("Công thức", "H2X"), P("L_i = -log( exp(score(q_i,p_i+)) / ( exp(score(q_i,p_i+)) + sum_j exp(score(q_i,p_j-)) ) )", "CodeX"), P("Cách hiểu", "H2X"), P("Loss phạt model khi positive score không cao hơn negative scores. Backpropagation cập nhật cả question encoder và passage encoder.", "BodyX")]
S.append(PageBreak())

# 7 code mapping
S += [P("7. Liên hệ với code trong project", "H1X"), P("File: src/training/negative_sampling.py", "H2X"), P("sample_bm25_hard_negatives(...) lọc các positive passage ID khỏi danh sách BM25 candidates, sau đó lấy những candidate còn lại làm hard negative.", "BodyX"), P("File: src/training/loss.py", "H2X"), P("in_batch_nll_loss(scores) tạo labels [0, ..., B-1] và áp dụng cross-entropy trên ma trận B x B. nll_loss_with_hard_negative(scores, hard_scores) nối thêm một cột hard-negative score.", "BodyX"), P("File: src/training/trainer.py", "H2X"), P("train_step() encode question, positive passage và optional hard negative; tính dot product; tính loss; gọi backward(); rồi optimizer.step(). fit() lặp qua nhiều epoch và lưu checkpoint.", "BodyX"), P("Format của một batch", "H2X"), P("{<br/>  question_inputs: {input_ids, attention_mask},<br/>  positive_passage_inputs: {input_ids, attention_mask},<br/>  hard_negative_inputs: {input_ids, attention_mask}<br/>}", "CodeX"), P("Shape cần nhớ", "H2X"), P("question embeddings: [B, D]<br/>passage embeddings: [B, D]<br/>scores: [B, B]<br/>hard scores: [B]<br/>final logits: [B, B + 1]", "CodeX")]
S.append(PageBreak())

# 8 board and QA
S += [P("8. Cách trình bày trên bảng", "H1X"), P("Vẽ theo thứ tự:", "H2X"), P("1. Hai encoder: Question Encoder và Passage Encoder.<br/>2. q và p trở thành dense vectors.<br/>3. Dot product tạo relevance score.<br/>4. Một batch tạo ma trận B x B.<br/>5. Đường chéo là positive, ngoài đường chéo là in-batch negatives.<br/>6. Thêm một cột BM25 hard negative.<br/>7. Cross-entropy loss và backpropagation.", "BodyX"), P("Câu nói ngắn bằng tiếng Anh", "H2X"), P("DPR uses two independent encoders to map questions and passages into dense vectors. In-batch negatives are obtained by reusing positive passages from other questions in the same batch. BM25 hard negatives are highly ranked but incorrect passages. The loss increases the positive score and decreases all negative scores.", "BodyX"), P("Câu hỏi thường gặp", "H2X"), P("Why not use only random negatives? -> They are often too easy.<br/>Why use in-batch negatives? -> They reuse existing computations and provide B - 1 negatives per question.<br/>What if a hard negative is actually relevant? -> It becomes a false negative, so positive passage IDs must be filtered.<br/>What does training learn? -> It learns encoder parameters, not the final answer text.", "BodyX"), P("Checklist hiểu bài", "H2X"), P("- Biết phân biệt p+ và p-.<br/>- Vẽ được ma trận B x B.<br/>- Giải thích được vì sao đường chéo là positive.<br/>- Giải thích được BM25 hard negative.<br/>- Nhớ được final logits có shape B x (B + 1).<br/>- Nói được training cập nhật encoder bằng backpropagation.", "BodyX")]

doc.build(S)
print(OUTPUT)
