import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://admin:password@cluster.mongodb.net/travel_social?retryWrites=true&w=majority")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
      # Tên collection trong MongoDB
    PLACES_COLLECTION = os.getenv("PLACES_COLLECTION", "Location")
    USERS_COLLECTION = os.getenv("USERS_COLLECTION", "users")
    RATINGS_COLLECTION = os.getenv("RATINGS_COLLECTION", "Review")
    EVENTS_COLLECTION = os.getenv("EVENTS_COLLECTION", "events")
    
    # Đường dẫn lưu trữ dữ liệu local
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    # Đảm bảo các thư mục dữ liệu tồn tại
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Prompts
    SYSTEM_PROMPT = """Bạn là trợ lý AI giúp trích xuất thông tin từ câu hỏi 
    để tìm địa điểm du lịch. Bạn sẽ phân tích yêu cầu và trả về kết quả 
    dựa trên dữ liệu có sẵn."""