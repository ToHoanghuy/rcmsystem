"""
Module cung cấp các utility để xử lý dữ liệu sự kiện và cập nhật mô hình
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import defaultdict
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EventProcessor:
    """
    Xử lý sự kiện và cập nhật dữ liệu cho mô hình
    """
    def __init__(self, data_dir, ratings_file='location_ratings.csv', events_file='location_events.csv'):
        """
        Khởi tạo Event Processor
        
        Parameters:
        -----------
        data_dir : str
            Đường dẫn đến thư mục dữ liệu
        ratings_file : str
            Tên file ratings
        events_file : str
            Tên file events
        """
        self.data_dir = data_dir
        self.ratings_path = os.path.join(data_dir, 'raw', ratings_file)
        self.events_path = os.path.join(data_dir, 'raw', events_file)
        
        # Đảm bảo thư mục raw tồn tại
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        
        # Khởi tạo các dataframe
        self.ratings_df = self._load_ratings()
        self.events_df = self._load_events()
        self.implicit_ratings = defaultdict(lambda: defaultdict(float))
        self._last_processed_event_timestamp = None
        
    def _load_ratings(self):
        """
        Load dữ liệu ratings từ file CSV hoặc tạo DataFrame mới nếu không tồn tại
        """
        try:
            if os.path.exists(self.ratings_path):
                df = pd.read_csv(self.ratings_path)
                logger.info(f"Loaded {len(df)} ratings from {self.ratings_path}")
                return df
            else:
                logger.warning(f"Ratings file {self.ratings_path} not found, creating new DataFrame")
                return pd.DataFrame(columns=['user_id', 'location_id', 'rating', 'timestamp'])
        except Exception as e:
            logger.error(f"Error loading ratings data: {str(e)}")
            return pd.DataFrame(columns=['user_id', 'location_id', 'rating', 'timestamp'])
            
    def _load_events(self):
        """
        Load dữ liệu events từ file CSV hoặc tạo DataFrame mới nếu không tồn tại
        """
        try:
            if os.path.exists(self.events_path):
                df = pd.read_csv(self.events_path)
                logger.info(f"Loaded {len(df)} events from {self.events_path}")
                
                # Parse timestamp nếu có
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        # Lưu timestamp mới nhất để chỉ xử lý sự kiện mới
                        if len(df) > 0:
                            self._last_processed_event_timestamp = df['timestamp'].max()
                    except:
                        logger.warning("Could not parse timestamps in events file")
                
                return df
            else:
                logger.warning(f"Events file {self.events_path} not found, creating new DataFrame")
                return pd.DataFrame(columns=['user_id', 'event_type', 'location_id', 'timestamp', 'data'])
        except Exception as e:
            logger.error(f"Error loading events data: {str(e)}")
            return pd.DataFrame(columns=['user_id', 'event_type', 'location_id', 'timestamp', 'data'])
    
    def process_new_events(self):
        """
        Xử lý các sự kiện mới chưa được xử lý
        """
        try:
            # Tải lại dữ liệu sự kiện
            fresh_events = self._load_events()
            
            # Lọc các sự kiện mới nếu có timestamp cuối đã xử lý
            if self._last_processed_event_timestamp and 'timestamp' in fresh_events.columns:
                try:
                    fresh_events['timestamp'] = pd.to_datetime(fresh_events['timestamp'])
                    new_events = fresh_events[fresh_events['timestamp'] > self._last_processed_event_timestamp]
                    if len(new_events) == 0:
                        logger.info("No new events to process")
                        return False
                    logger.info(f"Processing {len(new_events)} new events")
                    self._process_events_batch(new_events)
                    self._last_processed_event_timestamp = fresh_events['timestamp'].max()
                except Exception as e:
                    logger.error(f"Error processing events by timestamp: {str(e)}")
                    # Fallback: xử lý tất cả sự kiện
                    self._process_events_batch(fresh_events)
            else:
                # Xử lý tất cả sự kiện
                logger.info(f"Processing all {len(fresh_events)} events")
                self._process_events_batch(fresh_events)
                
            # Cập nhật dữ liệu
            self._update_ratings_from_implicit()
            return True
        except Exception as e:
            logger.error(f"Error in process_new_events: {str(e)}")
            return False
    
    def _process_events_batch(self, events_df):
        """
        Xử lý một batch sự kiện và cập nhật implicit ratings
        """
        if len(events_df) == 0:
            return
            
        # Hệ số điểm cho mỗi loại sự kiện
        event_weights = {
            'view': 0.1,
            'click': 0.2,
            'search': 0.1,
            'favorite': 0.8,
            'book': 0.9,
            'rate': 1.0  # Rating có điểm rõ ràng, không dùng hệ số này
        }
        
        # Reset implicit ratings
        self.implicit_ratings = defaultdict(lambda: defaultdict(float))
        
        # Xử lý từng sự kiện và tích lũy điểm
        for _, event in events_df.iterrows():
            user_id = str(event['user_id'])
            location_id = str(event['location_id'])
            event_type = event['event_type']
            
            # Xử lý rating riêng
            if event_type == 'rate':
                try:
                    data = event['data']
                    # Trích xuất rating từ data
                    if isinstance(data, str):
                        try:
                            data_dict = json.loads(data)
                            if 'rating' in data_dict:
                                rating = float(data_dict['rating'])
                                self._update_explicit_rating(user_id, location_id, rating)
                        except json.JSONDecodeError:
                            pass
                except Exception as e:
                    logger.error(f"Error processing rate event: {str(e)}")
                continue
            
            # Cộng dồn điểm ngầm định cho các loại sự kiện khác
            if event_type in event_weights:
                weight = event_weights[event_type]
                self.implicit_ratings[user_id][location_id] += weight
    
    def _update_explicit_rating(self, user_id, location_id, rating):
        """
        Cập nhật rating tường minh
        """
        # Tìm kiếm rating hiện có
        mask = (self.ratings_df['user_id'] == user_id) & (self.ratings_df['location_id'] == location_id)
        if mask.any():
            # Cập nhật rating nếu đã tồn tại
            self.ratings_df.loc[mask, 'rating'] = rating
            self.ratings_df.loc[mask, 'timestamp'] = datetime.now().isoformat()
        else:
            # Thêm rating mới
            new_rating = pd.DataFrame({
                'user_id': [user_id],
                'location_id': [location_id],
                'rating': [rating],
                'timestamp': [datetime.now().isoformat()]
            })
            self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)
        
        # Lưu cập nhật vào file
        self._save_ratings()
    
    def _update_ratings_from_implicit(self):
        """
        Cập nhật ratings từ implicit feedback
        """
        if not self.implicit_ratings:
            return
            
        # Chuẩn hóa điểm ngầm định thành ratings
        now = datetime.now().isoformat()
        rows = []
        
        for user_id, locations in self.implicit_ratings.items():
            for location_id, score in locations.items():
                # Chuyển đổi điểm ngầm định thành rating (từ 0-1 thành 1-5)
                # Giới hạn điểm tối đa là 4.5 cho implicit feedback
                normalized_score = min(1.0 + 3.5 * score, 4.5)
                
                # Kiểm tra xem đã có rating rõ ràng chưa
                mask = (self.ratings_df['user_id'] == user_id) & (self.ratings_df['location_id'] == location_id)
                
                if not mask.any():
                    # Chỉ thêm rating mới nếu chưa có rating rõ ràng
                    rows.append({
                        'user_id': user_id,
                        'location_id': location_id,
                        'rating': normalized_score,
                        'timestamp': now,
                        'is_implicit': True  # Đánh dấu rating ngầm định
                    })
        
        if rows:
            # Thêm cột is_implicit nếu chưa có
            if 'is_implicit' not in self.ratings_df.columns:
                self.ratings_df['is_implicit'] = False
                
            # Thêm các ratings mới vào DataFrame
            new_ratings = pd.DataFrame(rows)
            self.ratings_df = pd.concat([self.ratings_df, new_ratings], ignore_index=True)
            
            # Lưu cập nhật vào file
            self._save_ratings()
    
    def _save_ratings(self):
        """
        Lưu dữ liệu ratings vào file CSV
        """
        try:
            self.ratings_df.to_csv(self.ratings_path, index=False)
            logger.info(f"Saved {len(self.ratings_df)} ratings to {self.ratings_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving ratings data: {str(e)}")
            return False
    
    def get_new_ratings(self, since_timestamp=None):
        """
        Lấy các ratings mới từ một thời điểm
        
        Parameters:
        -----------
        since_timestamp : str or datetime
            Thời điểm từ đó lấy ratings mới
            
        Returns:
        --------
        DataFrame
            DataFrame chứa các ratings mới
        """
        if since_timestamp is None:
            return self.ratings_df
            
        # Chuyển đổi timestamp thành datetime
        if isinstance(since_timestamp, str):
            since_timestamp = pd.to_datetime(since_timestamp)
            
        # Đảm bảo ratings_df có cột timestamp là datetime
        if 'timestamp' in self.ratings_df.columns and not pd.api.types.is_datetime64_dtype(self.ratings_df['timestamp']):
            try:
                self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'])
            except:
                logger.warning("Could not parse timestamps in ratings data")
                return self.ratings_df
                
        # Lọc ratings mới
        if 'timestamp' in self.ratings_df.columns:
            return self.ratings_df[self.ratings_df['timestamp'] > since_timestamp]
        else:
            return self.ratings_df
