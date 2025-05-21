"""
MongoDB Change Stream Monitor

Module này sử dụng MongoDB Change Streams để theo dõi các thay đổi trong collection 'places'
và tự động cập nhật mô hình gợi ý khi có dữ liệu mới.
"""

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from threading import Thread, Event
import time
import logging
import pandas as pd
from config.config import Config
from database.mongodb import MongoDB
import os
import json
import sys

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBChangeMonitor:
    """
    Class giám sát thay đổi trong MongoDB và kích hoạt cập nhật mô hình khi cần thiết    """
    def __init__(self, update_callback=None, debounce_seconds=300):
        """
        Khởi tạo monitor
        
        Parameters:
        -----------
        update_callback : callable
            Hàm callback sẽ được gọi khi cần cập nhật mô hình
        debounce_seconds : int
            Thời gian chờ giữa các lần cập nhật (để tránh cập nhật liên tục)
        """
        try:
            self.mongodb = MongoDB()
            self.client = self.mongodb.client
            self.db = self.mongodb.db
            self.places = self.db[Config.PLACES_COLLECTION]  # Collection locations
            self.update_callback = update_callback
            self.debounce_seconds = debounce_seconds
            self.monitor_thread = None
            self.stop_event = Event()
            self.last_update_time = 0
            logger.info(f"MongoDBChangeMonitor đã khởi tạo thành công, theo dõi collection {Config.PLACES_COLLECTION}")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo MongoDBChangeMonitor: {str(e)}")
            raise
            
    def _get_latest_data(self):
        """
        Lấy dữ liệu mới nhất từ MongoDB và lưu vào tệp CSV
        """
        try:
            # Sử dụng MongoDB utility để xuất dữ liệu
            locations_path = os.path.join(Config.RAW_DATA_DIR, 'locations.csv')
            result = self.mongodb.export_to_csv(output_path=locations_path)
            
            if not result:
                logger.warning("Không có dữ liệu địa điểm để xuất")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lấy và lưu dữ liệu từ MongoDB: {str(e)}")
            return False
    
    def _check_for_updates(self):
        """
        Kiểm tra xem có nên cập nhật mô hình hay không
        """
        current_time = time.time()
        if (current_time - self.last_update_time) > self.debounce_seconds:
            # Nếu đã đủ thời gian kể từ lần cập nhật trước
            if self._get_latest_data() and self.update_callback:
                try:
                    # Gọi callback để cập nhật mô hình
                    self.update_callback()
                    self.last_update_time = current_time
                    logger.info(f"Đã cập nhật mô hình thành công - {time.ctime(current_time)}")
                except Exception as e:
                    logger.error(f"Lỗi khi cập nhật mô hình: {str(e)}")
    def _monitor_changes(self):
        """
        Giám sát sự thay đổi trong collection places
        """
        try:            # Thiết lập Change Stream để theo dõi thay đổi
            pipeline = [
                {'$match': {'operationType': {'$in': ['insert', 'update', 'delete', 'replace']}}},
                {'$match': {'ns.coll': Config.PLACES_COLLECTION}}  # Sử dụng tên collection từ config (locations)
            ]
            
            with self.places.watch(pipeline=pipeline) as stream:
                logger.info(f"Bắt đầu giám sát thay đổi trong MongoDB collection '{Config.PLACES_COLLECTION}'")
                
                # Khởi tạo dữ liệu ban đầu (đảm bảo có dữ liệu để bắt đầu)
                self._get_latest_data()
                
                # Chờ đợi thay đổi
                while not self.stop_event.is_set():
                    try:
                        # Kiểm tra xem có sự kiện thay đổi không
                        if stream.alive:
                            change = stream.try_next()
                            if change is not None:
                                logger.info(f"Phát hiện thay đổi: {change['operationType']} - {change.get('documentKey', 'Unknown')}")
                                self._check_for_updates()
                        
                        # Ngủ 1 giây trước khi thử lại
                        time.sleep(1)
                    except PyMongoError as e:
                        logger.error(f"Lỗi MongoDB trong quá trình theo dõi: {str(e)}")
                        break
        except PyMongoError as e:
            logger.error(f"Lỗi MongoDB trong quá trình giám sát: {str(e)}")
        except Exception as e:
            logger.error(f"Lỗi không xác định trong quá trình giám sát MongoDB: {str(e)}")
        
        # Thử kết nối lại sau 5 giây nếu thread chưa bị dừng
        if not self.stop_event.is_set():
            logger.info("Đang thử kết nối lại sau 5 giây...")
            time.sleep(5)
            self._monitor_changes()
    
    def start(self):
        """
        Bắt đầu giám sát MongoDB một cách không đồng bộ
        """
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = Thread(target=self._monitor_changes, daemon=True)
            self.monitor_thread.start()
            logger.info("Đã bắt đầu giám sát MongoDB")
            return True
        return False
    
    def stop(self):
        """
        Dừng giám sát MongoDB
        """
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join(timeout=5)
            logger.info("Đã dừng giám sát MongoDB")
            return True
        return False
    
    def force_update(self):
        """
        Cưỡng chế cập nhật mô hình ngay lập tức
        """
        if self._get_latest_data() and self.update_callback:
            self.update_callback()
            self.last_update_time = time.time()
            logger.info("Đã cưỡng chế cập nhật mô hình")
            return True
        return False
