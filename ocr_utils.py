"""
OCR và Vision Captioning module
================================
Module này cung cấp khả năng trích xuất văn bản từ hình ảnh (OCR)
và tạo mô tả ngữ nghĩa cho hình ảnh kỹ thuật (Vision Captioning).

Các thành phần chính:
- OCREngine: Sử dụng PaddleOCR để nhận dạng văn bản tiếng Việt/Anh
- VisionCaptioner: Sử dụng Qwen2-VL để mô tả nội dung hình ảnh
- process_image(): Kết hợp cả OCR và Vision để trích xuất thông tin đầy đủ

Sử dụng:
    from ocr_utils import process_image
    result = process_image("path/to/image.png", lang="vi")
    # result = {"ocr_text": "...", "caption": "...", "combined": "..."}
"""

from paddleocr import PaddleOCR
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from typing import Dict, List
from datetime import datetime

from config import VISION_MODEL, DEBUG_OCR, DEBUG_VISION


def log_debug(flag: bool, prefix: str, message: str):
    """
    Hàm helper để log debug có điều kiện.

    Args:
        flag: Cờ debug (True để hiển thị log)
        prefix: Tiền tố cho log (ví dụ: "🔍 OCR")
        message: Nội dung log
    """
    if flag:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {prefix}: {message}")


class OCREngine:
    """
    PaddleOCR Engine - Nhận dạng văn bản từ hình ảnh.

    Sử dụng PaddleOCR 3.x với hỗ trợ tiếng Việt để trích xuất
    văn bản từ hình ảnh như sơ đồ mạch, bảng thanh ghi, chú thích.

    Singleton Pattern: Chỉ tạo một instance duy nhất để tiết kiệm bộ nhớ.

    Attributes:
        reader: PaddleOCR instance

    Example:
        engine = OCREngine()
        text = engine.extract_text("diagram.png")
    """
    _instance = None

    def __new__(cls):
        """Singleton pattern - đảm bảo chỉ có 1 instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_ocr()
        return cls._instance

    def _init_ocr(self):
        """
        Khởi tạo PaddleOCR engine.

        PaddleOCR 3.x tự động detect GPU và sử dụng các model
        PP-OCRv5 mới nhất cho tiếng Việt.
        """
        print("Initializing PaddleOCR...")
        try:
            # PaddleOCR 3.x: chỉ cần lang, các parameter khác đã deprecated
            self.reader = PaddleOCR(lang='vi')
            print("PaddleOCR ready")
        except Exception as e:
            print(f"PaddleOCR init error: {e}")
            self.reader = None

    def extract_text(self, image_path: str) -> str:
        """
        Trích xuất văn bản từ hình ảnh.

        Quy trình:
        1. Gọi PaddleOCR predict() để phát hiện và nhận dạng text
        2. Lọc các kết quả có confidence > 0.5
        3. Ghép các dòng text thành chuỗi

        Args:
            image_path: Đường dẫn đến file hình ảnh

        Returns:
            str: Văn bản được trích xuất, các dòng cách nhau bởi newline.
                 Trả về chuỗi rỗng nếu không tìm thấy text hoặc có lỗi.

        Example:
            text = engine.extract_text("circuit.png")
            # "VCC\nGND\nPin 1: TX\nPin 2: RX"
        """
        try:
            if self.reader is None:
                print("OCR reader not initialized")
                return ""

            log_debug(DEBUG_OCR, "🔍 OCR", f"Processing: {image_path}")

            # PaddleOCR 3.x: sử dụng predict() thay vì ocr()
            result = self.reader.predict(image_path)
            if not result:
                log_debug(DEBUG_OCR, "🔍 OCR", "No text detected")
                return ""

            lines = []
            # PaddleOCR 3.x trả về list of dicts hoặc list of lists
            for item in result:
                if isinstance(item, dict):
                    # Format mới: {'rec_texts': [...], 'rec_scores': [...]}
                    texts = item.get('rec_texts', [])
                    scores = item.get('rec_scores', [])
                    for text, score in zip(texts, scores):
                        if score > 0.5:
                            lines.append(text)
                            log_debug(DEBUG_OCR, "🔍 OCR", f"  [{score:.2f}] {text[:50]}...")
                elif isinstance(item, list):
                    # Format cũ: [[box, (text, conf)], ...]
                    for line in item:
                        if len(line) >= 2 and isinstance(line[1], tuple):
                            text = line[1][0]
                            conf = line[1][1]
                            if conf > 0.5:
                                lines.append(text)
                                log_debug(DEBUG_OCR, "🔍 OCR", f"  [{conf:.2f}] {text[:50]}...")

            result_text = "\n".join(lines)
            log_debug(DEBUG_OCR, "🔍 OCR", f"Extracted {len(lines)} lines, {len(result_text)} chars")

            return result_text
        except Exception as e:
            print(f"OCR error: {e}")
            return ""


class VisionCaptioner:
    """
    Vision Captioner - Tạo mô tả ngữ nghĩa cho hình ảnh kỹ thuật.

    Sử dụng Qwen2-VL-7B để phân tích và mô tả nội dung hình ảnh
    như sơ đồ mạch, timing diagram, flowchart, biểu đồ.

    Model được giữ trong memory sau lần load đầu tiên để tăng tốc
    các lần xử lý tiếp theo.

    Singleton Pattern: Chỉ tạo một instance duy nhất.

    Attributes:
        _model: Qwen2-VL model instance
        _processor: Qwen2-VL processor để xử lý input
        _loaded: Flag cho biết model đã được load chưa
        _disabled: Flag để disable khi không đủ VRAM

    Example:
        captioner = VisionCaptioner()
        caption = captioner.caption_image("schematic.png", lang="vi")
    """
    _instance = None
    _model = None
    _processor = None
    _loaded = False
    _disabled = False  # Disable khi không đủ VRAM

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _check_vram(self, required_gb: float = 10.0) -> bool:
        """
        Kiểm tra VRAM còn trống có đủ để load model không.

        Args:
            required_gb: Số GB VRAM cần thiết (mặc định 10GB)

        Returns:
            bool: True nếu đủ VRAM, False nếu không đủ
        """
        try:
            if torch.cuda.is_available():
                free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
                print(f"Available VRAM: {free_memory:.2f} GB, required: {required_gb} GB")
                return free_memory >= required_gb
            return False
        except:
            return False

    def load_model(self) -> bool:
        """
        Load Vision model vào memory.

        Quy trình:
        1. Kiểm tra nếu đã load -> return True
        2. Kiểm tra nếu đã disable (không đủ VRAM) -> return False
        3. Kiểm tra VRAM còn trống (cần ~16GB)
        4. Load model từ Hugging Face với float16 precision
        5. Load processor để xử lý input

        Returns:
            bool: True nếu load thành công, False nếu thất bại
        """
        if self._loaded:
            return True

        if self._disabled:
            return False

        # Kiểm tra VRAM (cần ~16GB cho Qwen2-VL-7B với float16)
        if not self._check_vram(16.0):
            print("WARNING: Not enough VRAM for Vision model. Skipping image captioning.")
            self._disabled = True
            return False

        try:
            print(f"Loading Vision model: {VISION_MODEL}")
            log_debug(DEBUG_VISION, "🖼️ Vision", "Loading Qwen2-VL-7B model...")

            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                VISION_MODEL,
                torch_dtype=torch.float16,  # Sử dụng FP16 để tiết kiệm VRAM
                device_map="auto",          # Tự động map lên GPU
                trust_remote_code=True
            )
            self._processor = AutoProcessor.from_pretrained(
                VISION_MODEL,
                trust_remote_code=True
            )
            self._loaded = True
            print("Vision model loaded and kept in memory")
            log_debug(DEBUG_VISION, "🖼️ Vision", "Model ready")
            return True
        except Exception as e:
            print(f"Failed to load Vision model: {e}")
            self._disabled = True
            return False

    def caption_image(self, image_path: str, lang: str = "en") -> str:
        """
        Tạo mô tả ngữ nghĩa cho hình ảnh kỹ thuật.

        Quy trình:
        1. Load model nếu chưa load
        2. Chọn prompt phù hợp theo ngôn ngữ (vi/en)
        3. Mở và convert ảnh sang RGB
        4. Chuẩn bị input với chat template
        5. Generate caption với model
        6. Giải phóng memory để tránh fragmentation

        Args:
            image_path: Đường dẫn đến file hình ảnh
            lang: Ngôn ngữ output ("vi" hoặc "en")

        Returns:
            str: Mô tả ngữ nghĩa của hình ảnh.
                 Trả về chuỗi rỗng nếu không load được model hoặc có lỗi.

        Example:
            caption = captioner.caption_image("circuit.png", lang="vi")
            # "Sơ đồ mạch điều khiển LED sử dụng transistor NPN..."
        """
        try:
            if not self.load_model():
                return ""  # Skip nếu không load được model

            log_debug(DEBUG_VISION, "🖼️ Vision", f"Processing: {image_path}")

            # Prompt được thiết kế cho tài liệu kỹ thuật embedded
            if lang == "vi":
                prompt = "Mô tả chi tiết hình ảnh kỹ thuật này, tập trung vào sơ đồ mạch, code, linh kiện, cấu hình chân, hoặc thông tin hệ thống nhúng."
            else:
                prompt = "Describe this technical image in detail, focusing on circuit diagrams, code, hardware components, pin configurations, or embedded systems information."

            # Mở ảnh và convert sang RGB
            image = Image.open(image_path).convert("RGB")

            # Chuẩn bị input theo format chat của Qwen2-VL
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize và chuyển lên GPU
            inputs = self._processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(self._model.device)

            # Generate caption (không dùng sampling để có kết quả nhất quán)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=256,  # Giới hạn độ dài output
                    do_sample=False      # Deterministic output
                )

            # Decode output (bỏ phần input)
            output_ids = output_ids[:, inputs.input_ids.shape[1]:]
            caption = self._processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]

            # Giải phóng memory sau mỗi caption để tránh fragmentation
            del inputs, output_ids
            torch.cuda.empty_cache()

            log_debug(DEBUG_VISION, "🖼️ Vision", f"Caption ({len(caption)} chars): {caption[:100]}...")

            return caption
        except Exception as e:
            print(f"Vision error: {e}")
            return ""

    def caption_batch(self, image_paths: List[str], lang: str = "en") -> List[str]:
        """
        Tạo caption cho nhiều ảnh (xử lý tuần tự).

        Args:
            image_paths: Danh sách đường dẫn đến các file ảnh
            lang: Ngôn ngữ output ("vi" hoặc "en")

        Returns:
            List[str]: Danh sách caption tương ứng với từng ảnh
        """
        captions = []
        for i, path in enumerate(image_paths):
            log_debug(DEBUG_VISION, "🖼️ Vision", f"Batch [{i+1}/{len(image_paths)}]: {path}")
            caption = self.caption_image(path, lang)
            captions.append(caption)
        return captions

    def unload_model(self):
        """
        Giải phóng VRAM bằng cách unload model.

        Sử dụng khi cần giải phóng VRAM cho các tác vụ khác.
        Sau khi unload, cần gọi load_model() lại để sử dụng.
        """
        if self._loaded:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            torch.cuda.empty_cache()
            print("Vision model unloaded")


def process_image(image_path: str, lang: str = "en") -> Dict:
    """
    Xử lý hình ảnh: kết hợp OCR và Vision captioning.

    Đây là hàm chính để trích xuất thông tin từ hình ảnh. Nó kết hợp:
    - OCR: Trích xuất văn bản có trong ảnh (chú thích, giá trị, label)
    - Vision: Tạo mô tả ngữ nghĩa về cấu trúc và ý nghĩa hình ảnh

    Quy trình:
    1. Gọi OCREngine để trích xuất text
    2. Gọi VisionCaptioner để tạo mô tả
    3. Kết hợp kết quả thành format thống nhất

    Args:
        image_path: Đường dẫn đến file hình ảnh
        lang: Ngôn ngữ cho Vision caption ("vi" hoặc "en")

    Returns:
        Dict với các key:
        - ocr_text: Văn bản từ OCR
        - caption: Mô tả từ Vision model
        - combined: Kết hợp cả hai theo format:
            [Image Description]
            {caption}

            [OCR Text]
            {ocr_text}

    Example:
        result = process_image("diagram.png", lang="vi")
        print(result["combined"])
    """
    log_debug(DEBUG_OCR or DEBUG_VISION, "📷 Image", f"Processing: {image_path}")

    # Bước 1: OCR - Trích xuất văn bản
    ocr_engine = OCREngine()
    ocr_text = ocr_engine.extract_text(image_path)

    # Bước 2: Vision - Tạo mô tả ngữ nghĩa
    captioner = VisionCaptioner()
    caption = captioner.caption_image(image_path, lang)
    # Không unload model nữa - giữ trong memory để xử lý nhanh hơn

    # Bước 3: Kết hợp kết quả
    if ocr_text and caption:
        combined = f"[Image Description]\n{caption}\n\n[OCR Text]\n{ocr_text}"
    elif caption:
        combined = f"[Image Description]\n{caption}"
    elif ocr_text:
        combined = f"[OCR Text]\n{ocr_text}"
    else:
        combined = "[No content extracted from image]"

    log_debug(DEBUG_OCR or DEBUG_VISION, "📷 Image",
              f"Result: OCR={len(ocr_text)} chars, Caption={len(caption)} chars")

    return {
        "ocr_text": ocr_text,
        "caption": caption,
        "combined": combined
    }


def get_ocr_engine() -> OCREngine:
    """
    Factory function để lấy OCREngine instance.

    Returns:
        OCREngine: Singleton instance của OCREngine
    """
    return OCREngine()


def get_vision_captioner() -> VisionCaptioner:
    """
    Factory function để lấy VisionCaptioner instance.

    Returns:
        VisionCaptioner: Singleton instance của VisionCaptioner
    """
    return VisionCaptioner()
