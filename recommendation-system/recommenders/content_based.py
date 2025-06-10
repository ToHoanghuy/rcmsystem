from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import os
import pickle

def extract_features(product_details, save_vectorizer_scaler=True, vectorizer_path='tfidf_vectorizer.pkl', scaler_path='num_scaler.pkl'):
    """
    Trích xuất và kết hợp các đặc trưng khác nhau từ dữ liệu sản phẩm, cải tiến xử lý text, categorical, numerical, giảm sparse, hỗ trợ đa ngôn ngữ.
    Luôn thêm col_missing cho numerical, tự động lưu vectorizer & scaler, xử lý sâu slug/category_id.
    """
    import re
    import unicodedata
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import RobustScaler, LabelEncoder
    from sklearn.decomposition import TruncatedSVD
    from langdetect import detect
    import numpy as np
    import pandas as pd
    import json
    # 1. Chuẩn hóa text, tạo cột description nếu cần
    def normalize_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\x00', '')
        text = re.sub(r'[^\w\sÀ-ỹ-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Tạo description nếu chưa có hoặc rỗng
    if 'description' not in product_details.columns or product_details['description'].fillna('').str.strip().eq('').all():
        text_cols = [col for col in product_details.columns if col != 'product_id' and product_details[col].dtype == 'object']
        if text_cols:
            product_details['description'] = product_details[text_cols].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)
        else:
            product_details['description'] = product_details['product_id'].astype(str)
    product_details['description'] = product_details['description'].fillna('').apply(normalize_text)

    # Xử lý slug sâu hơn: tách slug thành các token phân tầng
    if 'slug' in product_details.columns:
        product_details['slug'] = product_details['slug'].fillna('').apply(normalize_text)
        # Tách slug thành các phần (nếu dạng a-b-c)
        product_details['slug_tokens'] = product_details['slug'].apply(lambda x: x.split('-') if '-' in x else [x])
        # Tạo feature phân tầng slug depth
        product_details['slug_depth'] = product_details['slug_tokens'].apply(len)
        # Tạo feature slug_prefix (phân tầng đầu tiên)
        product_details['slug_prefix'] = product_details['slug_tokens'].apply(lambda x: x[0] if len(x) > 0 else '')

    # Xử lý category.id sâu hơn
    if 'category' in product_details.columns:
        def extract_cat_id(val):
            if isinstance(val, dict):
                return val.get('id', '')
            if isinstance(val, str):
                try:
                    obj = json.loads(val)
                    return obj.get('id', '')
                except:
                    return val
            return ''
        product_details['category_id'] = product_details['category'].apply(extract_cat_id)
        # Tách category_id phân tầng nếu dạng phân cấp (vd: cat1/cat2)
        product_details['category_main'] = product_details['category_id'].apply(lambda x: x.split('/')[0] if '/' in x else x)
        product_details['category_depth'] = product_details['category_id'].apply(lambda x: len(x.split('/')) if isinstance(x, str) and '/' in x else 1)

    # 2. Xử lý text với TF-IDF nâng cao, hỗ trợ đa ngôn ngữ
    descriptions = product_details['description'].tolist()
    try:
        langs = [detect(desc) if desc.strip() else 'unknown' for desc in descriptions]
        main_lang = max(set(langs), key=langs.count)
    except Exception:
        main_lang = 'unknown'
    from sklearn.feature_extraction import text as sk_text
    stop_words = sk_text.ENGLISH_STOP_WORDS
    if main_lang == 'vi':
        try:
            from underthesea import word_tokenize
            def vi_tokenizer(text):
                return word_tokenize(text, format="text").split()
            tokenizer = vi_tokenizer
        except ImportError:
            tokenizer = None
    else:
        tokenizer = None
    tfidf = TfidfVectorizer(
        max_features=300,
        stop_words=stop_words,
        ngram_range=(1,2),
        tokenizer=tokenizer
    )
    text_features = tfidf.fit_transform(descriptions)
    # Lưu vectorizer nếu cần
    if save_vectorizer_scaler:
        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(tfidf, f)
        except Exception as e:
            print(f"[WARN] Could not save tfidf vectorizer: {e}")
    # Giảm chiều nếu quá sparse
    if text_features.shape[1] > 50:
        svd = TruncatedSVD(n_components=50, random_state=42)
        text_features = svd.fit_transform(text_features)
        text_df = pd.DataFrame(text_features, index=product_details.index, columns=[f'txt_{i}' for i in range(50)])
    else:
        text_df = pd.DataFrame(text_features.toarray(), index=product_details.index, columns=[f'txt_{i}' for i in range(text_features.shape[1])])

    # 3. Xử lý categorical: thêm category_id, slug, slug_prefix, category_main nếu có
    cat_columns = [col for col in product_details.columns if product_details[col].dtype == 'object' and col not in ['product_id', 'description']]
    for extra_col in ['category_id', 'slug', 'slug_prefix', 'category_main']:
        if extra_col in product_details.columns:
            cat_columns.append(extra_col)
    cat_columns = list(set(cat_columns))
    categorical_features = pd.DataFrame(index=product_details.index)
    for col in cat_columns:
        nunique = product_details[col].nunique()
        if nunique > 20:
            freq = product_details[col].value_counts(normalize=True)
            categorical_features[f'{col}_freq'] = product_details[col].map(freq)
        else:
            dummies = pd.get_dummies(product_details[col], prefix=col)
            categorical_features = pd.concat([categorical_features, dummies], axis=1)

    # 4. Xử lý numerical: điền thiếu bằng median, thêm col_missing (binary 0/1), robust scaler
    num_columns = [col for col in product_details.select_dtypes(include=['float64', 'int64']).columns if col != 'product_id']
    numerical_features = pd.DataFrame(index=product_details.index)
    missing_features = pd.DataFrame(index=product_details.index)
    if num_columns:
        for col in num_columns:
            median = product_details[col].median()
            numerical_features[col] = product_details[col].fillna(median)
            # Luôn thêm cột missing (0/1)
            missing_features[f'{col}_missing'] = product_details[col].isnull().astype(int)
        scaler = RobustScaler()
        numerical_features = pd.DataFrame(scaler.fit_transform(numerical_features), columns=num_columns, index=product_details.index)
        # Lưu scaler nếu cần
        if save_vectorizer_scaler:
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            except Exception as e:
                print(f"[WARN] Could not save scaler: {e}")

    # 5. Feature tổng hợp: số lượng ảnh, độ dài mô tả, v.v.
    if 'image' in product_details.columns:
        categorical_features['num_images'] = product_details['image'].apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0)
    categorical_features['desc_len'] = product_details['description'].apply(lambda x: len(x.split()))

    # 6. Kết hợp tất cả
    all_features = pd.concat([categorical_features, missing_features, numerical_features, text_df], axis=1)
    if all_features.shape[1] == 0:
        all_features['dummy'] = 1.0
    print(f"[IMPROVED] Feature extraction complete. Total features: {all_features.shape[1]}")
    return all_features

def train_content_based_model(product_details):
    """
    Huấn luyện mô hình Content-Based Filtering
    
    Parameters:
    -----------
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
        
    Returns:
    --------
    similarity_matrix: numpy.ndarray
        Ma trận tương đồng giữa các sản phẩm
    index_to_id: dict
        Dictionary ánh xạ từ index trong ma trận tương đồng đến product_id
    id_to_index: dict
        Dictionary ánh xạ từ product_id đến index trong ma trận tương đồng
    """
    try:
        # FORCE VALID: Đảm bảo DataFrame không rỗng
        if product_details.empty:
            print("[DEBUG] WARNING: Empty product_details DataFrame. Cannot train model.")
            # Tạo ma trận giả với 1 dòng (sản phẩm)
            dummy_id = "dummy_product_1"
            dummy_df = pd.DataFrame({
                'product_id': [dummy_id],
                'description': ['Dummy product for empty dataset']
            })
            product_details = dummy_df
            print("[DEBUG] Created dummy product for empty dataset")
            
    # Kiểm tra product_id có phải là cột không
        if 'product_id' not in product_details.columns:
            raise ValueError("product_details must contain 'product_id' column")
            
        # Đảm bảo không có duplicates trong product_id
        if product_details['product_id'].duplicated().any():
            print("[DEBUG] WARNING: Duplicate product_ids found. Keeping first occurrence.")
            product_details = product_details.drop_duplicates(subset='product_id', keep='first')
            
        # Trích xuất đặc trưng
        features = extract_features(product_details)
        
        # Kiểm tra và xử lý NaN values trước khi tính toán tương đồng
        if features.isnull().values.any():
            print(f"[DEBUG] Warning: Found {features.isnull().sum().sum()} NaN values in features. Filling with 0...")
            features = features.fillna(0)
        
        # FORCE VALID: Đảm bảo features không rỗng
        if features.shape[1] == 0:
            print("[DEBUG] WARNING: No features extracted. Creating dummy feature.")
            features['dummy'] = 1.0
            features['dummy'] = 1.0
            
        # Tính toán ma trận tương đồng
        print("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(features)
        
        # Lưu trữ ánh xạ giữa index và product_id cho việc tra cứu sau này
        index_to_id = dict(enumerate(product_details['product_id'].values))
        id_to_index = {id_val: idx for idx, id_val in index_to_id.items()}
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        return similarity_matrix, index_to_id, id_to_index
    
    except Exception as e:
        print(f"Error training Content-Based Filtering model: {str(e)}")
        # Return valid but empty results as fallback
        dummy_id = "dummy_product"
        similarity_matrix = np.array([[1.0]])
        index_to_id = {0: dummy_id}
        id_to_index = {dummy_id: 0}
        return similarity_matrix, index_to_id, id_to_index

def get_similar_products(product_id, similarity_matrix, id_to_index, index_to_id, product_details, top_n=10):
    """
    Lấy các sản phẩm tương tự dựa trên content-based filtering
    
    Parameters:
    -----------
    product_id: int
        ID của sản phẩm cần tìm các sản phẩm tương tự
    similarity_matrix: numpy.ndarray
        Ma trận tương đồng giữa các sản phẩm
    id_to_index: dict
        Dictionary ánh xạ từ product_id đến index trong ma trận tương đồng
    index_to_id: dict
        Dictionary ánh xạ từ index trong ma trận tương đồng đến product_id
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    top_n: int
        Số lượng sản phẩm tương tự cần trả về
        
    Returns:
    --------
    similar_products: DataFrame
        DataFrame chứa thông tin về các sản phẩm tương tự
    """
    if product_id not in id_to_index:
        print(f"Product ID {product_id} not found in the database")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu không tìm thấy sản phẩm
    
    # Lấy index của sản phẩm
    idx = id_to_index[product_id]
    
    # Lấy điểm tương đồng với các sản phẩm khác
    similarity_scores = similarity_matrix[idx]
    
    # Lấy index của top N sản phẩm tương tự (bỏ qua sản phẩm hiện tại)
    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    
    # Kiểm tra xem có sản phẩm tương tự không
    if len(similar_indices) == 0:
        print(f"No similar products found for product ID {product_id}")
        return pd.DataFrame()
    
    # Lấy ID của các sản phẩm tương tự
    similar_product_ids = [index_to_id[i] for i in similar_indices]
    
    # Lấy thông tin chi tiết
    similar_products = product_details[product_details['product_id'].isin(similar_product_ids)].copy()
    
    # Kiểm tra xem có tìm được thông tin chi tiết không
    if similar_products.empty:
        print(f"No product details found for similar products of ID {product_id}")
        return pd.DataFrame()
    
    # Thêm điểm tương đồng vào kết quả
    similarity_dict = {index_to_id[i]: similarity_scores[i] for i in similar_indices}
    similar_products['similarity_score'] = similar_products['product_id'].map(similarity_dict)
    
    # Sắp xếp theo điểm tương đồng giảm dần
    similar_products = similar_products.sort_values('similarity_score', ascending=False)
    
    return similar_products
    

def recommend_content_based(user_id, ratings, product_details, top_n=10):
    """
    Gợi ý sản phẩm dựa trên Content-Based Filtering
    
    Parameters:
    -----------
    user_id: int
        ID của người dùng cần gợi ý
    ratings: DataFrame
        DataFrame chứa dữ liệu đánh giá của người dùng với các cột 'user_id', 'product_id', 'rating'
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    top_n: int
        Số lượng sản phẩm gợi ý cần trả về
        
    Returns:
    --------
    recommendations: DataFrame
        DataFrame chứa thông tin về các sản phẩm được gợi ý
    """
    print(f"Generating content-based recommendations for user {user_id}")
    
    # Kiểm tra xem người dùng có xếp hạng nào không
    if ratings is None or ratings.empty:
        # Nếu không có đánh giá nào, trả về các sản phẩm phổ biến nhất
        if 'rating' in product_details.columns:
            return product_details.sort_values(by='rating', ascending=False).head(top_n)
        else:
            return product_details.head(top_n)
    
    # Trích xuất đặc trưng từ sản phẩm
    features = extract_features(product_details)
    
    # Tính ma trận tương đồng
    similarity_matrix = cosine_similarity(features)
    
    # Ánh xạ giữa index và product_id
    index_to_id = dict(enumerate(product_details['product_id'].values))
    id_to_index = {id_val: idx for idx, id_val in index_to_id.items()}
    
    # Lấy các sản phẩm mà người dùng đã đánh giá cao
    user_rated_products = ratings[ratings['rating'] >= 4].sort_values(by='rating', ascending=False)
    
    # Nếu người dùng không có đánh giá cao nào, lấy tất cả đánh giá của họ
    if user_rated_products.empty:
        user_rated_products = ratings
    
    # Giới hạn số lượng sản phẩm đã đánh giá để xem xét (để tránh thời gian tính toán quá lâu)
    user_top_products = user_rated_products.head(5)
    
    # Tìm sản phẩm tương tự cho mỗi sản phẩm đã được đánh giá cao
    similar_items = pd.DataFrame()
    
    for _, row in user_top_products.iterrows():
        product_id = row['product_id']
        rating = row['rating']
        
        try:
            # Kiểm tra xem product_id có trong danh sách sản phẩm không
            if product_id not in id_to_index:
                continue
                
            # Lấy sản phẩm tương tự
            idx = id_to_index[product_id]
            similarity_scores = similarity_matrix[idx]
            
            # Chuyển thành DataFrame
            similar_df = pd.DataFrame({
                'product_id': [index_to_id[i] for i in range(len(similarity_scores))],
                'similarity': similarity_scores,
                'user_rating': rating
            })
            
            # Loại bỏ sản phẩm gốc
            similar_df = similar_df[similar_df['product_id'] != product_id]
            
            # Tính điểm nội dung = similarity * user_rating
            similar_df['content_score'] = similar_df['similarity'] * similar_df['user_rating'] / 5.0
            
            similar_items = pd.concat([similar_items, similar_df])
            
        except Exception as e:
            print(f"Error finding similar items for product {product_id}: {e}")
    
    # Loại bỏ các sản phẩm trùng lặp (lấy điểm cao nhất)
    if not similar_items.empty:
        recommendations = similar_items.groupby('product_id')['content_score'].max().reset_index()
        
        # Loại bỏ các sản phẩm người dùng đã đánh giá
        rated_product_ids = set(ratings['product_id'])
        recommendations = recommendations[~recommendations['product_id'].isin(rated_product_ids)]
        
        # Sắp xếp theo điểm giảm dần
        recommendations = recommendations.sort_values(by='content_score', ascending=False).head(top_n)
        
        # Thêm thông tin chi tiết về sản phẩm
        if not recommendations.empty:
            recommendations = pd.merge(recommendations, product_details, on='product_id', how='left')
            
        return recommendations
    else:
        # Nếu không tìm thấy sản phẩm tương tự, trả về các sản phẩm phổ biến nhất
        if 'rating' in product_details.columns:
            return product_details.sort_values(by='rating', ascending=False).head(top_n)
        else:
            return product_details.head(top_n)