"""
Script chạy thử hệ thống đề xuất với dữ liệu location
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Thêm thư mục gốc vào path để có thể import các module trong dự án
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from scripts.integrate_locations import integrate_location_data
from scripts.train_model import train_all_location_models
from scripts.generate_recommendations import (
    generate_location_recommendations, 
    generate_personalized_location_recommendations,
    generate_switching_location_recommendations
)

def run_location_recommendations_demo():
    """
    Chạy thử đề xuất với dữ liệu location
    """
    print("\n" + "="*80)
    print("DEMO HỆ THỐNG ĐỀ XUẤT ĐỊA ĐIỂM DU LỊCH")
    print("="*80)
    
    # Bước 1: Tích hợp dữ liệu location
    print("\n-- BƯỚC 1: TÍCH HỢP DỮ LIỆU LOCATION --")
    try:
        integrated_data_path = integrate_location_data()
        if not integrated_data_path or not os.path.exists(integrated_data_path):
            print("Không thể tích hợp dữ liệu location. Thoát chương trình.")
            return
    except Exception as e:
        print(f"Lỗi khi tích hợp dữ liệu: {str(e)}")
        return
    
    # Bước 2: Chuẩn bị dữ liệu cho huấn luyện mô hình
    print("\n-- BƯỚC 2: CHUẨN BỊ DỮ LIỆU --")
    
    try:
        # Đường dẫn đến các file dữ liệu
        locations_path = root_dir / "data" / "raw" / "locations.csv"
        ratings_path = root_dir / "data" / "raw" / "location_ratings.csv"
        events_path = root_dir / "data" / "raw" / "location_events.csv"
        
        # Kiểm tra xem các file có tồn tại không
        if not os.path.exists(locations_path):
            print(f"Không tìm thấy file dữ liệu location: {locations_path}")
            return
        
        if not os.path.exists(ratings_path):
            print(f"Không tìm thấy file dữ liệu đánh giá: {ratings_path}")
            return
        
        # Đọc dữ liệu
        locations_df = pd.read_csv(locations_path)
        ratings_df = pd.read_csv(ratings_path)
        events_df = pd.read_csv(events_path) if os.path.exists(events_path) else None
        
        print(f"Đọc được {len(locations_df)} địa điểm")
        print(f"Đọc được {len(ratings_df)} đánh giá")
        if events_df is not None:
            print(f"Đọc được {len(events_df)} sự kiện")
        else:
            print("Không có dữ liệu sự kiện")
            
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {str(e)}")
        return
    
    # Bước 3: Huấn luyện mô hình
    print("\n-- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH --")
    
    try:
        collaborative_model, content_model_data = train_all_location_models(
            ratings_df, locations_df, events_df
        )
        print("Đã huấn luyện mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        return
    
    # Bước 4: Chạy thử đề xuất
    print("\n-- BƯỚC 4: CHẠY THỬ ĐỀ XUẤT --")
    
    # Tạo user_id ngẫu nhiên
    test_user_ids = [1, 3, 5]  # Chọn một số user_id từ dữ liệu đánh giá
    
    # Tạo vị trí user giả định
    user_locations = [
        {"latitude": 21.0278, "longitude": 105.8342},  # Hà Nội
        {"latitude": 10.8231, "longitude": 106.6297},  # TP.HCM
        {"latitude": 16.0544, "longitude": 108.2022},  # Đà Nẵng
    ]
    
    # Ghép user_id và vị trí
    test_scenarios = list(zip(test_user_ids, user_locations))
    
    for user_id, user_location in test_scenarios:
        print(f"\nĐề xuất cho người dùng {user_id} tại vị trí: {user_location}")
        
        # Phương pháp 1: Đề xuất location với ngữ cảnh
        print("\n1. ĐỀ XUẤT ĐỊA ĐIỂM THEO NGỮ CẢNH:")
        context_recommendations = generate_location_recommendations(
            user_id=user_id,
            collaborative_model=collaborative_model,
            content_model_data=content_model_data,
            product_details=locations_df,
            user_location=user_location,
            top_n=5
        )
        
        if not context_recommendations.empty:
            print_recommendations(context_recommendations)
        else:
            print("Không có đề xuất theo ngữ cảnh")
        
        # Phương pháp 2: Đề xuất location được cá nhân hóa
        print("\n2. ĐỀ XUẤT ĐỊA ĐIỂM CÁ NHÂN HÓA:")
        personalized_recommendations = generate_personalized_location_recommendations(
            user_id=user_id,
            collaborative_model=collaborative_model,
            content_model_data=content_model_data,
            product_details=locations_df,
            top_n=5
        )
        
        if not personalized_recommendations.empty:
            print_recommendations(personalized_recommendations)
        else:
            print("Không có đề xuất cá nhân hóa")
        
        # Phương pháp 3: Đề xuất location với chuyển đổi
        print("\n3. ĐỀ XUẤT ĐỊA ĐIỂM CƠ SỞ:")
        switching_recommendations, rec_type = generate_switching_location_recommendations(
            user_id=user_id,
            collaborative_model=collaborative_model,
            content_model_data=content_model_data,
            product_details=locations_df,
            top_n=5
        )
        
        if not switching_recommendations.empty:
            print(f"Phương pháp sử dụng: {rec_type}")
            print_recommendations(switching_recommendations)
        else:
            print("Không có đề xuất cơ sở")
    
    print("\n" + "="*80)
    print("HOÀN THÀNH DEMO")
    print("="*80 + "\n")

def print_recommendations(recommendations_df):
    """In kết quả đề xuất theo định dạng dễ đọc"""
    
    # Các cột cần hiển thị
    display_columns = ['name', 'category', 'province', 'address', 'rating']
    score_column = next((col for col in recommendations_df.columns if 'score' in col.lower()), None)
    
    if score_column:
        for _, row in recommendations_df.iterrows():
            print(f"- {row['name']} ({row['category']})")
            print(f"  Địa chỉ: {row['address']}, {row['province']}")
            print(f"  Đánh giá: {row['rating']}/5")
            print(f"  Điểm đề xuất: {row[score_column]:.3f}")
    else:
        for _, row in recommendations_df.iterrows():
            print(f"- {row['name']} ({row['category']})")
            print(f"  Địa chỉ: {row['address']}, {row['province']}")
            print(f"  Đánh giá: {row['rating']}/5")

if __name__ == "__main__":
    run_location_recommendations_demo()
