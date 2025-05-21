"""
Script để kiểm tra kết nối với MongoDB của backend
"""
import os
import sys
import pandas as pd

# Thêm thư mục gốc vào path để import các module
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from database.mongodb import MongoDB
from config.config import Config

def test_mongodb_connection():
    """
    Kiểm tra kết nối với MongoDB và truy xuất dữ liệu
    """
    try:
        print("=== KIỂM TRA KẾT NỐI MONGODB ===")
        print("Đang kết nối đến MongoDB...")
        mongodb = MongoDB()
        
        # Kiểm tra kết nối
        if not mongodb.test_connection():
            print("❌ Không thể kết nối đến MongoDB!")
            return False
            
        print("✅ Kết nối MongoDB thành công!")
        
        # Kiểm tra truy cập collection places
        try:
            places_collection = mongodb.places
            places_count = places_collection.count_documents({})
            print(f"✅ Tìm thấy collection '{Config.PLACES_COLLECTION}' (locations) với {places_count} địa điểm")
            
            # Lấy mẫu dữ liệu
            if places_count > 0:
                places_sample = list(places_collection.find({}).limit(3))
                print("\nMẫu dữ liệu địa điểm:")
                for i, place in enumerate(places_sample, 1):
                    print(f"\n--- Địa điểm {i} ---")
                    if '_id' in place:
                        place['_id'] = str(place['_id'])  # Chuyển ObjectId sang string
                    
                    # Hiển thị các trường quan trọng
                    important_fields = ['_id', 'placeName', 'placeAddress', 'placeProvince', 'placeRating', 'placePrice']
                    for field in important_fields:
                        if field in place:
                            print(f"{field}: {place[field]}")
                    
                    # Hiển thị số lượng trường còn lại
                    other_fields = [f for f in place.keys() if f not in important_fields]
                    print(f"... và {len(other_fields)} trường khác")
                
                # Kiểm tra mapping trường với hệ thống đề xuất
                print("\nKiểm tra mapping trường dữ liệu:")
                required_fields = ['product_id', 'name', 'address', 'description', 'price', 'rating', 'image']
                mapping = {
                    '_id': 'product_id', 
                    'placeName': 'name',
                    'placeAddress': 'address',
                    'placeDescription': 'description',
                    'placePrice': 'price',
                    'placeRating': 'rating',
                    'placeImage': 'image'
                }
                
                for backend_field, recommendation_field in mapping.items():
                    has_field = any(backend_field in place for place in places_sample)
                    status = "✅" if has_field else "❌"
                    print(f"{status} {backend_field} → {recommendation_field}")
                
                # Xuất dữ liệu mẫu thành CSV
                print("\nĐang xuất dữ liệu ra CSV...")
                if mongodb.export_to_csv():
                    print(f"✅ Đã xuất dữ liệu thành công ra {os.path.join(Config.RAW_DATA_DIR, 'locations.csv')}")
                else:
                    print("❌ Xuất dữ liệu thất bại")
                
            return True
                
        except Exception as e:
            print(f"❌ Lỗi khi truy cập collection: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return False

if __name__ == "__main__":
    test_mongodb_connection()
