"""
Test script cho hàm clean_mongodb_json nâng cao
"""
import json
import pandas as pd
import numpy as np
from utils.mongodb_json_cleaner import clean_mongodb_json

# Test case 1: Giá trị NaN trong nhiều dạng khác nhau
test_data_nan = {
    "__v": 0.0,
    "_id": "671a0ba18d9d58d19294038e",
    "caategory": float('nan'),
    "rating": np.nan,
    "minPrice": "NaN",
    "longitude": np.float64('nan'),
    "numberOfRating": None
}

# Test case 2: Chuỗi JSON lồng nhau (trường category)
test_data_json_string = {
    "category": "{\"id\": \"homestay\", \"cateName\": \"Quán cafe\"}",
    "metadata": "{invalid json string}",
    "tags": "[\"sang trọng\", \"đẹp\"]"
}

# Test case 3: Mảng trong chuỗi (trường image)
test_data_array_string = {
    "image": "[{'url': 'https://example.com/img1.jpg', 'publicId': 'id1'}, {'url': 'https://example.com/img2.jpg', 'publicId': 'id2'}]",
    "invalid_array": "[{test: invalid json}]"
}

# Test case 4: ObjectId trong chuỗi
test_data_object_id = {
    "image": "[{'url': 'https://example.com/img1.jpg', '_id': ObjectId('6773e64087afd958c3bbb33f')}]"
}

# Test case 5: Dữ liệu thực tế
test_real_data = {
    "__v": float('nan'),
    "_id": "6704f3650722c4f99305dc5f",
    "address": "789 Đường Lê Văn Sỹ, Hà Nội",
    "caategory": float('nan'),
    "category": "{\"id\": \"hotel\", \"name\": \"Khách sạn\"}",
    "dateCreated": "2024-10-08T10:00:00",
    "description": "Quán cafe với không gian sáng tạo, lý tưởng cho làm việc và gặp gỡ bạn bè.",
    "image": "[{'url': 'https://res.cloudinary.com/dzy4gcw1k/image/upload/v1735585136/travel-social/file-1735585106495-pew-nguyen-3sXuWhKZVeg-unsplash.jpg.jpg', 'publicId': 'travel-social/file-1735585106495-pew-nguyen-3sXuWhKZVeg-unsplash.jpg'}]",
    "latitude": 21.028511,
    "longitude": 105.804817,
    "minPrice": 200000.0,
    "name": "Quán Cafe Sáng Tạo",
    "numberOfRating": 11.0,
    "ownerId": "6704df9d3d81e2f463846c27",
    "product_id": "6704f3650722c4f99305dc5f",
    "province": "Hà Nội",
    "rating": 3.7,
    "slug": "quan-cafe-sang-tao-789-uong-le-van-sy-ha-noi",
    "status": "active"
}

# Chạy các test
print("=== TEST 1: Xử lý giá trị NaN ===")
cleaned_nan = clean_mongodb_json(test_data_nan)
print(json.dumps(cleaned_nan, indent=2, ensure_ascii=False))

print("\n=== TEST 2: Xử lý chuỗi JSON lồng nhau ===")
cleaned_json_string = clean_mongodb_json(test_data_json_string)
print(json.dumps(cleaned_json_string, indent=2, ensure_ascii=False))

print("\n=== TEST 3: Xử lý chuỗi mảng ===")
cleaned_array_string = clean_mongodb_json(test_data_array_string)
print(json.dumps(cleaned_array_string, indent=2, ensure_ascii=False))

print("\n=== TEST 4: Xử lý ObjectId trong chuỗi ===")
cleaned_object_id = clean_mongodb_json(test_data_object_id)
print(json.dumps(cleaned_object_id, indent=2, ensure_ascii=False))

print("\n=== TEST 5: Xử lý dữ liệu thực tế ===")
cleaned_real_data = clean_mongodb_json(test_real_data)
print(json.dumps(cleaned_real_data, indent=2, ensure_ascii=False))

print("\nTest hoàn tất!")
