from pymongo import MongoClient
from config.config import Config
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDB:
    def __init__(self):
        """
        Khởi tạo kết nối MongoDB sử dụng URI từ Config hoặc .env
        """
        load_dotenv()  # Tải biến môi trường từ .env
        
        # Lấy URI từ Config
        mongodb_uri = Config.MONGODB_URI
        
        if not mongodb_uri:
            raise ValueError("MONGODB_URI không được cấu hình trong Config hoặc .env")
            
        # Kết nối MongoDB
        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.get_database()  # Tự động lấy database từ connection string
            self.places = self.db[Config.PLACES_COLLECTION]  # Collection locations
            logger.info(f"Kết nối MongoDB thành công tới collection {Config.PLACES_COLLECTION}")
        except Exception as e:
            logger.error(f"Lỗi kết nối MongoDB: {str(e)}")
            raise
    
    def test_connection(self):
        """
        Kiểm tra kết nối MongoDB
        """
        try:
            # Ping để kiểm tra kết nối
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"Không thể kết nối MongoDB: {str(e)}")
            return False
    
    def query_places(self, filters):
        """
        Tìm kiếm địa điểm theo bộ lọc
        """
        mongo_filter = {
            "province": {"$regex": filters['province'], "$options": "i"},
            "services": {"$all": filters.get("services", [])},
            "capacity": {"$gte": filters.get("min_capacity", 0)}
        }
        return list(self.places.find(mongo_filter))

    def add_place(self, place_data):
        """
        Thêm địa điểm mới vào database
        """
        return self.places.insert_one(place_data)
        
    def get_all_places(self):
        """
        Lấy tất cả địa điểm từ database
        """
        return list(self.places.find({}))
        
    def export_to_csv(self, output_path=None):
        """
        Xuất tất cả địa điểm ra file CSV
        """
        import pandas as pd
        
        if not output_path:
            output_path = os.path.join(Config.RAW_DATA_DIR, 'locations.csv')
            
        try:
            # Lấy tất cả địa điểm
            places = self.get_all_places()
            
            if not places:
                logger.warning("Không có dữ liệu địa điểm để xuất")
                return False
                
            # Chuyển đổi _id thành string để có thể lưu vào CSV
            for place in places:
                if '_id' in place:
                    place['_id'] = str(place['_id'])
            
            # Chuyển thành DataFrame và lưu vào CSV
            df = pd.DataFrame(places)
            
            # Đảm bảo có cột product_id (cần thiết cho hệ thống đề xuất)
            if 'product_id' not in df.columns:
                if '_id' in df.columns:
                    df['product_id'] = df['_id']  # Sử dụng _id làm product_id
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Lưu vào CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Đã xuất {len(df)} địa điểm ra {output_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xuất dữ liệu ra CSV: {str(e)}")
            return False