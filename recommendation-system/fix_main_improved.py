"""
Hướng dẫn sửa lỗi và cập nhật chức năng hiển thị thông tin đầy đủ của location trong API

Bạn cần sửa hai lỗi chính trong main.py:

1. SỬA LỖI ĐỊNH DẠNG (SyntaxError): Điều chỉnh định dạng của các hàm:
   - Cần có dòng trống giữa docstring và code đầu tiên
   - Sửa lại định dạng của hàm process_advanced_collaborative_recommendation

2. CÁCH HIỂN THỊ THÔNG TIN ĐẦY ĐỦ: Cập nhật phần else trong hàm process_advanced_collaborative_recommendation để hiển thị đầy đủ thông tin location:

```python
def process_advanced_collaborative_recommendation(user_id, product_id=None):
"""
"""
    # Đảm bảo user_id có định dạng nhất quán (string hoặc số)
    if user_id is not None and not isinstance(user_id, (int, str)):
        user_id = str(user_id)
        
    if product_id is None:
        # Gợi ý tất cả địa điểm cho người dùng
        location_ids = location_details['product_id'].tolist()
        recommendations = []
        
        # Đảm bảo location_ids có định dạng nhất quán
        location_ids = [str(id) if isinstance(id, str) else id for id in location_ids]
        
        # Sử dụng batch predict để tăng hiệu suất
        batch_size = min(100, len(location_ids))
        predictions = batch_predict(advanced_collaborative_model, [user_id], location_ids, batch_size)
        
        if user_id in predictions:
            user_predictions = predictions[user_id]
            for pid, pred_rating in user_predictions.items():
                # Lấy thông tin chi tiết của location từ location_details
                location_info = location_details[location_details['product_id'] == pid].to_dict('records')
                
                if location_info:
                    # Kết hợp thông tin chi tiết với dự đoán rating
                    location_data = location_info[0].copy()
                    location_data["predicted_rating"] = pred_rating
                    recommendations.append(location_data)
                else:
                    # Nếu không tìm thấy thông tin chi tiết, chỉ trả về ID và rating
                    recommendations.append({
                        "product_id": pid,
                        "predicted_rating": pred_rating
                    })
            
            # Sắp xếp theo điểm dự đoán giảm dần
            recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations
    else:
        # Dự đoán cho một địa điểm cụ thể
        try:
            # Thử dự đoán với mô hình advanced_collaborative
            prediction = advanced_collaborative_model.predict(uid=user_id, iid=product_id)
            pred_rating = prediction.est
            
            # Lấy thông tin chi tiết của location
            location_info = location_details[location_details['product_id'] == product_id].to_dict('records')
            if location_info:
                # Kết hợp thông tin chi tiết với dự đoán rating
                result = location_info[0].copy()
                result["predicted_rating"] = pred_rating
                return result
            return {"product_id": product_id, "predicted_rating": pred_rating}
        except:
            # Nếu không được, sử dụng xử lý cold-start
            pred_rating = predict_for_cold_start(user_id, product_id, advanced_collaborative_model, content_model_data[0], location_details)
            
            # Lấy thông tin chi tiết của location
            location_info = location_details[location_details['product_id'] == product_id].to_dict('records')
            if location_info:
                # Kết hợp thông tin chi tiết với dự đoán rating
                result = location_info[0].copy()
                result["predicted_rating"] = pred_rating
                return result
            return {"product_id": product_id, "predicted_rating": pred_rating}
```

CÁCH SỬA KHÁC:
Đây là phương pháp thay thế: Tạo một phiên bản mới của file main.py. Sao chép toàn bộ nội dung và thay thế phần không hoạt động bằng đoạn code đã được sửa từ hướng dẫn trên.
"""

def make_advanced_collaborative_template():
    return """def process_advanced_collaborative_recommendation(user_id, product_id=None):
    \"\"\"
    Xử lý gợi ý dựa trên Collaborative Filtering nâng cao (SVD++)
    \"\"\"
    # Đảm bảo user_id có định dạng nhất quán (string hoặc số)
    if user_id is not None and not isinstance(user_id, (int, str)):
        user_id = str(user_id)
        
    if product_id is None:
        # Gợi ý tất cả địa điểm cho người dùng
        location_ids = location_details['product_id'].tolist()
        recommendations = []
        
        # Đảm bảo location_ids có định dạng nhất quán
        location_ids = [str(id) if isinstance(id, str) else id for id in location_ids]
        
        # Sử dụng batch predict để tăng hiệu suất
        batch_size = min(100, len(location_ids))
        predictions = batch_predict(advanced_collaborative_model, [user_id], location_ids, batch_size)
        
        if user_id in predictions:
            user_predictions = predictions[user_id]
            for pid, pred_rating in user_predictions.items():
                # Lấy thông tin chi tiết của location từ location_details
                location_info = location_details[location_details['product_id'] == pid].to_dict('records')
                
                if location_info:
                    # Kết hợp thông tin chi tiết với dự đoán rating
                    location_data = location_info[0].copy()
                    location_data["predicted_rating"] = pred_rating
                    recommendations.append(location_data)
                else:
                    # Nếu không tìm thấy thông tin chi tiết, chỉ trả về ID và rating
                    recommendations.append({
                        "product_id": pid,
                        "predicted_rating": pred_rating
                    })
            
            # Sắp xếp theo điểm dự đoán giảm dần
            recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations
    else:
        # Dự đoán cho một địa điểm cụ thể
        try:
            # Thử dự đoán với mô hình advanced_collaborative
            prediction = advanced_collaborative_model.predict(uid=user_id, iid=product_id)
            pred_rating = prediction.est
            
            # Lấy thông tin chi tiết của location
            location_info = location_details[location_details['product_id'] == product_id].to_dict('records')
            if location_info:
                # Kết hợp thông tin chi tiết với dự đoán rating
                result = location_info[0].copy()
                result["predicted_rating"] = pred_rating
                return result
            return {"product_id": product_id, "predicted_rating": pred_rating}
        except:
            # Nếu không được, sử dụng xử lý cold-start
            pred_rating = predict_for_cold_start(user_id, product_id, advanced_collaborative_model, content_model_data[0], location_details)
            
            # Lấy thông tin chi tiết của location
            location_info = location_details[location_details['product_id'] == product_id].to_dict('records')
            if location_info:
                # Kết hợp thông tin chi tiết với dự đoán rating
                result = location_info[0].copy()
                result["predicted_rating"] = pred_rating
                return result
            return {"product_id": product_id, "predicted_rating": pred_rating}
"""

if __name__ == "__main__":
    print("\n------- HƯỚNG DẪN SỬA LỖI TRONG FILE MAIN.PY -------\n")
    print("Lỗi hiện tại: 'SyntaxError: invalid syntax' và API chỉ hiển thị product_id và predicted_rating")
    print("Để sửa lỗi và hiển thị đầy đủ thông tin location, cần thay thế hàm process_advanced_collaborative_recommendation với đoạn code sau:\n")
    print(make_advanced_collaborative_template())
    print("\nLưu ý: Đảm bảo định dạng đúng của docstring và không gian trắng giữa các phần của code.")
    print("\nKhi đã sửa thành công, API sẽ trả về thông tin đầy đủ của location bao gồm tên, địa chỉ, và các chi tiết khác.")
