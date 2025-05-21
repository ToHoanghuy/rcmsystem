"""
Data Importer Service

Module này hỗ trợ nhập dữ liệu vào MongoDB và tự động kích hoạt cập nhật mô hình gợi ý.
"""

import pandas as pd
import json
import os
import logging
import sys

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongodb import MongoDB
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataImporter:
    """
    Class quản lý việc nhập dữ liệu vào MongoDB và cập nhật mô hình đề xuất
    """
    def __init__(self, update_callback=None):
        """
        Khởi tạo DataImporter
        
        Parameters:
        -----------
        update_callback : callable
            Hàm callback sẽ được gọi khi cập nhật dữ liệu thành công
        """
        self.mongodb = MongoDB()
        self.update_callback = update_callback
    
    def import_csv(self, csv_path, collection_name='places'):
        """
        Nhập dữ liệu từ file CSV vào MongoDB
        
        Parameters:
        -----------
        csv_path : str
            Đường dẫn đến file CSV
        collection_name : str
            Tên collection trong MongoDB
            
        Returns:
        --------
        dict
            Kết quả của quá trình nhập dữ liệu
        """
        try:
            # Đọc dữ liệu từ file CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Đọc thành công file {csv_path}: {len(df)} dòng")
            
            # Chuyển đổi DataFrame thành danh sách các dictionary
            records = df.to_dict(orient='records')
            
            # Xác định collection
            collection = getattr(self.mongodb.db, collection_name)
            
            # Xóa dữ liệu cũ nếu cần
            if collection.count_documents({}) > 0:
                result = collection.delete_many({})
                logger.info(f"Đã xóa {result.deleted_count} documents từ collection {collection_name}")
            
            # Thêm dữ liệu mới
            result = collection.insert_many(records)
            inserted_count = len(result.inserted_ids)
            logger.info(f"Đã thêm {inserted_count} documents vào collection {collection_name}")
            
            # Lưu một bản sao vào thư mục data/raw
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
            os.makedirs(data_dir, exist_ok=True)
            
            if collection_name == 'places':
                output_path = os.path.join(data_dir, 'locations.csv')
            else:
                output_path = os.path.join(data_dir, f'{collection_name}.csv')
            
            df.to_csv(output_path, index=False)
            logger.info(f"Đã lưu dữ liệu vào {output_path}")
            
            # Gọi callback nếu có
            if self.update_callback:
                try:
                    self.update_callback()
                    logger.info("Đã gọi callback cập nhật mô hình")
                except Exception as e:
                    logger.error(f"Lỗi khi gọi callback: {str(e)}")
            
            return {
                "status": "success",
                "inserted_count": inserted_count,
                "collection": collection_name,
                "backup_path": output_path
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi nhập dữ liệu từ CSV: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def import_json(self, json_path, collection_name='places'):
        """
        Nhập dữ liệu từ file JSON vào MongoDB
        
        Parameters:
        -----------
        json_path : str
            Đường dẫn đến file JSON
        collection_name : str
            Tên collection trong MongoDB
            
        Returns:
        --------
        dict
            Kết quả của quá trình nhập dữ liệu
        """
        try:
            # Đọc dữ liệu từ file JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Nếu JSON là một đối tượng duy nhất, chuyển thành list
                records = [data]
            else:
                raise ValueError("Cấu trúc JSON không hỗ trợ. Vui lòng cung cấp một mảng hoặc đối tượng.")
            
            logger.info(f"Đọc thành công file {json_path}: {len(records)} records")
            
            # Xác định collection
            collection = getattr(self.mongodb.db, collection_name)
            
            # Xóa dữ liệu cũ nếu cần
            if collection.count_documents({}) > 0:
                result = collection.delete_many({})
                logger.info(f"Đã xóa {result.deleted_count} documents từ collection {collection_name}")
            
            # Thêm dữ liệu mới
            result = collection.insert_many(records)
            inserted_count = len(result.inserted_ids)
            logger.info(f"Đã thêm {inserted_count} documents vào collection {collection_name}")
            
            # Lưu một bản sao dưới dạng CSV trong thư mục data/raw
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
            os.makedirs(data_dir, exist_ok=True)
            
            # Chuyển đổi sang DataFrame để lưu dưới dạng CSV
            df = pd.DataFrame(records)
            
            if collection_name == 'places':
                output_path = os.path.join(data_dir, 'locations.csv')
            else:
                output_path = os.path.join(data_dir, f'{collection_name}.csv')
            
            df.to_csv(output_path, index=False)
            logger.info(f"Đã lưu dữ liệu vào {output_path}")
            
            # Gọi callback nếu có
            if self.update_callback:
                try:
                    self.update_callback()
                    logger.info("Đã gọi callback cập nhật mô hình")
                except Exception as e:
                    logger.error(f"Lỗi khi gọi callback: {str(e)}")
            
            return {
                "status": "success",
                "inserted_count": inserted_count,
                "collection": collection_name,
                "backup_path": output_path
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi nhập dữ liệu từ JSON: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
