"""OCR và Vision Captioning module"""
"""ocr_utils.py"""
from paddleocr import PaddleOCR
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from typing import Dict, List
from config import VISION_MODEL


class OCREngine:
    """PaddleOCR engine - hỗ trợ tiếng Việt và tiếng Anh"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reader = None
            cls._instance._initialized = False
            cls._instance._init_ocr()
        return cls._instance

    def _init_ocr(self):
        """Initialize PaddleOCR - auto-detects GPU/CPU based on PaddlePaddle installation"""
        try:
            print("Initializing PaddleOCR...")
            # Note: use_gpu parameter was removed in PaddleOCR v2.6+
            # GPU/CPU is now auto-detected based on PaddlePaddle installation
            self.reader = PaddleOCR(
                use_angle_cls=True,  # Detect rotated text
                lang='vi',  # Vietnamese (includes English)
                show_log=False
            )
            self._initialized = True
            print("PaddleOCR ready")
        except Exception as e:
            print(f"Warning: Failed to initialize PaddleOCR: {e}")
            self.reader = None
            self._initialized = False

    def extract_text(self, image_path: str) -> str:
        """Extract text từ ảnh"""
        if not self._initialized or self.reader is None:
            print("OCR error: PaddleOCR not initialized")
            return ""

        try:
            result = self.reader.ocr(image_path, cls=True)
            if not result or not result[0]:
                return ""

            lines = []
            for line in result[0]:
                text = line[1][0]  # text content
                conf = line[1][1]  # confidence
                if conf > 0.5:
                    lines.append(text)

            return "\n".join(lines)
        except Exception as e:
            print(f"OCR error: {e}")
            return ""


class VisionCaptioner:
    """Qwen2-VL captioner - giữ model trong memory"""
    _instance = None
    _model = None
    _processor = None
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Load model nếu chưa load"""
        if self._loaded:
            return
        
        print(f"Loading Vision model: {VISION_MODEL}")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            VISION_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(
            VISION_MODEL,
            trust_remote_code=True
        )
        self._loaded = True
        print("Vision model loaded and kept in memory")
    
    def caption_image(self, image_path: str, lang: str = "en") -> str:
        """Generate caption cho ảnh kỹ thuật"""
        try:
            self.load_model()
            
            if lang == "vi":
                prompt = "Mô tả chi tiết hình ảnh kỹ thuật này, tập trung vào sơ đồ mạch, code, linh kiện, cấu hình chân, hoặc thông tin hệ thống nhúng."
            else:
                prompt = "Describe this technical image in detail, focusing on circuit diagrams, code, hardware components, pin configurations, or embedded systems information."
            
            image = Image.open(image_path).convert("RGB")
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(self._model.device)
            
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=256, do_sample=False
                )
            
            output_ids = output_ids[:, inputs.input_ids.shape[1]:]
            caption = self._processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            
            return caption
        except Exception as e:
            print(f"Vision error: {e}")
            return ""
    
    def caption_batch(self, image_paths: List[str], lang: str = "en") -> List[str]:
        """Batch caption nhiều ảnh cùng lúc"""
        captions = []
        for path in image_paths:
            caption = self.caption_image(path, lang)
            captions.append(caption)
        return captions
    
    def unload_model(self):
        """Giải phóng VRAM khi cần"""
        if self._loaded:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            torch.cuda.empty_cache()
            print("Vision model unloaded")


def process_image(image_path: str, lang: str = "en") -> Dict:
    """Process ảnh: OCR + Vision caption"""
    ocr_engine = OCREngine()
    ocr_text = ocr_engine.extract_text(image_path)
    
    captioner = VisionCaptioner()
    caption = captioner.caption_image(image_path, lang)
    # Không unload model nữa - giữ trong memory
    
    if ocr_text and caption:
        combined = f"[Image Description]\n{caption}\n\n[OCR Text]\n{ocr_text}"
    elif caption:
        combined = f"[Image Description]\n{caption}"
    elif ocr_text:
        combined = f"[OCR Text]\n{ocr_text}"
    else:
        combined = "[No content extracted from image]"
    
    return {
        "ocr_text": ocr_text,
        "caption": caption,
        "combined": combined
    }


def get_ocr_engine() -> OCREngine:
    return OCREngine()


def get_vision_captioner() -> VisionCaptioner:
    return VisionCaptioner()