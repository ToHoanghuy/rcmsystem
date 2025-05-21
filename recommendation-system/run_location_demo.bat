@echo off
REM Script chạy demo hệ thống đề xuất địa điểm
REM Sử dụng dữ liệu location thực tế

echo ===== BẮT ĐẦU CHẠY DEMO ĐỀ XUẤT ĐỊA ĐIỂM =====
echo.

REM Đường dẫn đến thư mục gốc của dự án
cd /d %~dp0

REM Kiểm tra và tạo môi trường ảo Python nếu chưa có
if not exist venv (
    echo Tạo môi trường ảo Python...
    python -m venv venv
)

REM Kích hoạt môi trường ảo
call venv\Scripts\activate

REM Cài đặt các gói phụ thuộc
echo Cài đặt các gói phụ thuộc...
pip install -r requirements.txt

REM Thêm các gói mới cần thiết cho hệ thống đề xuất địa điểm
pip install geopy

REM Chạy script demo
echo.
echo Chạy demo hệ thống đề xuất địa điểm...
python -m utils.run_location_demo

REM Kết thúc
echo.
echo ===== KẾT THÚC DEMO =====
pause
