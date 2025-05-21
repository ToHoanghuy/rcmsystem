#!/bin/bash

# Script chạy demo hệ thống đề xuất địa điểm
# Sử dụng dữ liệu location thực tế

echo "===== BẮT ĐẦU CHẠY DEMO ĐỀ XUẤT ĐỊA ĐIỂM ====="
echo ""

# Đường dẫn đến thư mục gốc của dự án
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$PROJECT_DIR"

# Kiểm tra và tạo môi trường ảo Python nếu chưa có
if [ ! -d "venv" ]; then
    echo "Tạo môi trường ảo Python..."
    python -m venv venv
fi

# Kích hoạt môi trường ảo
source venv/bin/activate

# Cài đặt các gói phụ thuộc
echo "Cài đặt các gói phụ thuộc..."
pip install -r requirements.txt

# Thêm các gói mới cần thiết cho hệ thống đề xuất địa điểm
pip install geopy

# Chạy script demo
echo ""
echo "Chạy demo hệ thống đề xuất địa điểm..."
python -m utils.run_location_demo

# Kết thúc
echo ""
echo "===== KẾT THÚC DEMO ====="
