"""
Script để tích hợp dữ liệu location thực tế vào hệ thống đề xuất.

Script này sẽ:
1. Đọc dữ liệu location từ file CSV mới
2. Chuyển đổi định dạng dữ liệu nếu cần thiết
3. Thay thế hoặc tích hợp với dữ liệu sản phẩm hiện có
4. Lưu kết quả vào thư mục processed
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào path để có thể import các module trong dự án
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from preprocess.preprocess import preprocess_data

def integrate_location_data():
    """
    Tích hợp dữ liệu location vào hệ thống đề xuất
    """
    print("Bắt đầu tích hợp dữ liệu location...")
    
    # Đường dẫn đến các file dữ liệu
    locations_path = root_dir / "data" / "raw" / "locations.csv"
    products_path = root_dir / "data" / "raw" / "products.csv"
    events_path = root_dir / "data" / "raw" / "events.csv"
    output_path = root_dir / "data" / "processed" / "location_integrated_data.csv"
    
    # Kiểm tra xem file locations.csv có tồn tại không
    if not os.path.exists(locations_path):
        print(f"Không tìm thấy file {locations_path}")
        return
    
    # Đọc dữ liệu location
    print(f"Đọc dữ liệu location từ {locations_path}...")
    locations_df = pd.read_csv(locations_path)
    
    # In thông tin về dữ liệu location
    print(f"Đã đọc {len(locations_df)} bản ghi location:")
    print(locations_df.head())
    print("\nCấu trúc dữ liệu:")
    print(locations_df.dtypes)
    
    # Đọc dữ liệu sự kiện 
    if os.path.exists(events_path):
        print(f"Đọc dữ liệu sự kiện từ {events_path}...")
        events_df = pd.read_csv(events_path)
        
        # Cập nhật dữ liệu sự kiện để sử dụng các location_id mới
        print("Cập nhật dữ liệu sự kiện để sử dụng các location_id...")
        
        # Giả sử chúng ta có mapping từ product_id cũ đến location_id mới (trong thực tế, cần xây dựng logic mapping)
        # Ở đây chúng ta sẽ tạo một số sự kiện mẫu cho các location mới
        new_events = []
        
        # Tạo một số sự kiện mẫu cho mỗi location
        for location_id in locations_df['product_id'].unique():
            # Giả lập một số người dùng (1-10) tương tác với location này
            for user_id in range(1, 11):
                # Tạo sự kiện view
                new_events.append({
                    'user_id': user_id,
                    'product_id': location_id,
                    'event_type': 'view',
                    'timestamp': '2025-05-01T10:00:00'
                })
                
                # Tạo sự kiện add_to_cart cho một số người dùng
                if user_id % 3 == 0:
                    new_events.append({
                        'user_id': user_id,
                        'product_id': location_id,
                        'event_type': 'add_to_cart',
                        'timestamp': '2025-05-01T10:05:00'
                    })
                
                # Tạo sự kiện purchase cho một số người dùng
                if user_id % 5 == 0:
                    new_events.append({
                        'user_id': user_id,
                        'product_id': location_id,
                        'event_type': 'purchase',
                        'timestamp': '2025-05-01T10:10:00'
                    })
        
        # Tạo DataFrame mới từ các sự kiện mẫu
        new_events_df = pd.DataFrame(new_events)
        
        # Kết hợp với dữ liệu sự kiện hiện có (nếu có)
        if not events_df.empty:
            events_df = pd.concat([events_df, new_events_df], ignore_index=True)
        else:
            events_df = new_events_df
        
        # Lưu dữ liệu sự kiện đã cập nhật
        events_df.to_csv(root_dir / "data" / "raw" / "location_events.csv", index=False)
        print(f"Đã tạo và lưu {len(new_events)} sự kiện mới cho các location.")
    
    # Tạo dữ liệu đánh giá (ratings) từ thông tin rating có sẵn trong dữ liệu location
    print("Tạo dữ liệu đánh giá từ thông tin rating có sẵn...")
    ratings = []
    
    # Với mỗi location, tạo một số đánh giá dựa trên rating có sẵn
    for _, location in locations_df.iterrows():
        location_id = location['product_id']
        avg_rating = location['rating']
        
        if pd.notna(avg_rating) and avg_rating > 0:
            # Xác định số lượng đánh giá dựa trên mức rating
            num_ratings = max(5, int(avg_rating * 3))  # Ít nhất 5 đánh giá, tăng theo rating
            
            # Tạo các đánh giá xung quanh giá trị trung bình
            import numpy as np
            for user_id in range(1, num_ratings + 1):
                # Tạo rating xung quanh giá trị trung bình với độ lệch nhỏ
                rating_value = min(5, max(1, np.random.normal(avg_rating, 0.5)))
                ratings.append({
                    'user_id': user_id,
                    'product_id': location_id,
                    'rating': round(rating_value, 1)
                })
    
    # Tạo DataFrame đánh giá
    ratings_df = pd.DataFrame(ratings)
    
    # Lưu dữ liệu đánh giá
    ratings_path = root_dir / "data" / "raw" / "location_ratings.csv"
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Đã tạo và lưu {len(ratings_df)} đánh giá cho các location.")
    
    # Tiền xử lý dữ liệu tích hợp (sử dụng hàm preprocess_data hiện có)
    try:
        print("Tiền xử lý và tích hợp dữ liệu...")
        
        # Gọi hàm preprocess_data (giả sử đã có trong preprocess module)
        # Nếu hàm không tương thích, cần điều chỉnh hoặc viết mã xử lý riêng
        integrated_data = preprocess_data(
            ratings_path=ratings_path,
            products_path=locations_path,
            events_path=root_dir / "data" / "raw" / "location_events.csv" if os.path.exists(root_dir / "data" / "raw" / "location_events.csv") else None
        )
        
        # Lưu dữ liệu đã tích hợp
        integrated_data.to_csv(output_path, index=False)
        print(f"Đã lưu dữ liệu tích hợp tại {output_path}")
        
    except Exception as e:
        print(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
        
        # Phương án dự phòng: sao chép dữ liệu location làm output
        print("Sử dụng phương án dự phòng: sao chép dữ liệu location làm output")
        locations_df.to_csv(output_path, index=False)
    
    print("Hoàn thành tích hợp dữ liệu location.")
    return output_path

if __name__ == "__main__":
    integrate_location_data()
