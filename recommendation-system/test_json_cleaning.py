"""
Test script for the clean_json_data function in json_utils.py
This script tests the ability to handle problematic JSON data from MongoDB
"""
import json
from utils.json_utils import clean_json_data

# Example data with common MongoDB export issues
test_data = {
    "__v": 0.0,
    "_id": "6773e64087afd958c3bbb33e",
    "address": "44B Phó Đức Chính, Phường Thắng Tam, Thành phố Vũng Tàu, Bà Rịa - Vũng Tàu",
    "caategory": float('nan'),  # NaN value
    "category": "{\"id\": \"hotel\"}",  # JSON string instead of object
    "dateCreated": "Tue, 31 Dec 2024 12:40:32 GMT",
    "description": "Khách sạn\n  Địa chỉ: 44B Phó Đức Chính, Phường Thắng Tam, Thành phố Vũng Tàu, Bà Rịa - Vũng Tàu\n Số phòng: 21\n Điện thoại cố định: 0254 3616 333\n Fax: 0254 3600 767\n Email: haianhkhachsan@gmail.com\n Thông tin do Cơ quan nhà nước quản lý",
    # ObjectId in string format
    "image": "[{'url': 'https://res.cloudinary.com/dzy4gcw1k/image/upload/v1735648858/travel-social/file-1735648826795-4_%20%281%29.jpg.jpg', 'publicId': 'travel-social/file-1735648826795-4_ (1).jpg', '_id': ObjectId('6773e64087afd958c3bbb33f')}, {'url': 'https://res.cloudinary.com/dzy4gcw1k/image/upload/v1735648856/travel-social/file-1735648826800-5_.jpg.jpg', 'publicId': 'travel-social/file-1735648826800-5_.jpg', '_id': ObjectId('6773e64087afd958c3bbb340')}]",
    "latitude": float('nan'),  # NaN value
    "longitude": float('nan'),  # NaN value
    "minPrice": float('nan'),  # NaN value
    "name": "Khách sạn Hải Anh 3",
    "numberOfRating": 0.0,
    "ownerId": "671a02c2c0202050e0969548",
    "product_id": "6773e64087afd958c3bbb33e",
    "province": None,
    "rating": 0.0,
    "slug": None,
    "status": "active"
}

# Test the clean_json_data function
cleaned_data = clean_json_data(test_data)

# Print the result in a formatted way
print("=== ORIGINAL DATA ===")
print(test_data)
print("\n=== CLEANED DATA ===")
print(json.dumps(cleaned_data, indent=2, ensure_ascii=False))

# Test with nested lists and dictionaries with Unicode characters
nested_test = {
    "locations": [
        {
            "name": "Nhà hàng Hoàng Gia",
            "description": "Nhà hàng chuyên món ăn Việt Nam truyền thống",
            "ratings": [4.5, 4.8, float('nan'), 4.7],
            "details": {
                "address": "123 Lê Lợi, Quận 1, TP.HCM",
                "phones": ["0123456789", None, "0987654321"],
            }
        },
        {
            "name": "Khách sạn Đông Phương",
            "tags": "[\"sang trọng\", \"view đẹp\", \"phục vụ tốt\"]",
            "metadata": "{\"stars\": 5, \"rooms\": 120}"
        }
    ]
}

# Test the clean_json_data function with nested data
cleaned_nested = clean_json_data(nested_test)

print("\n=== NESTED ORIGINAL DATA ===")
print(nested_test)
print("\n=== CLEANED NESTED DATA ===")
print(json.dumps(cleaned_nested, indent=2, ensure_ascii=False))

print("\nTest completed successfully!")
