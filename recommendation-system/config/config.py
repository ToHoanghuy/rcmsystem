import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    
    # Prompts
    SYSTEM_PROMPT = """Bạn là trợ lý AI giúp trích xuất thông tin từ câu hỏi 
    để tìm địa điểm du lịch. Bạn sẽ phân tích yêu cầu và trả về kết quả 
    dựa trên dữ liệu có sẵn."""