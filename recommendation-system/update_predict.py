import re
import os

def update_model_predict_calls(file_path):
    """
    Updates all collaborative_model.predict calls in the file to handle string IDs
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Pattern to match simple predict calls without try/except
    pattern1 = r'(\s+)([a-zA-Z_]+)\s*=\s*collaborative_model\.predict\(uid=([a-zA-Z_0-9]+),\s*iid=([a-zA-Z_0-9]+)\)'
    
    # Replace with try/except version
    replacement1 = r'''\1try:
\1    try:
\1        \2 = collaborative_model.predict(uid=\3, iid=\4)
\1    except (ValueError, TypeError) as e:
\1        # Nếu lỗi kiểu dữ liệu, thử với dạng chuỗi
\1        print(f"Converting IDs to string: {e}")
\1        \2 = collaborative_model.predict(uid=str(\3), iid=str(\4))
\1except Exception as e:
\1    print(f"Hybrid prediction failed for user {\3}, product {\4}: {e}")
\1    \2 = None  # or default value'''
    
    updated_content = re.sub(pattern1, replacement1, content)
    
    # Pattern for calls already in a try block
    pattern2 = r'(\s+)try:\s*\n\s*([a-zA-Z_]+)\s*=\s*collaborative_model\.predict\(uid=([a-zA-Z_0-9]+),\s*iid=([a-zA-Z_0-9]+)\)\.est\s*\n\s*except:'
    
    # Replace with nested try/except
    replacement2 = r'''\1try:
\1    try:
\1        \2 = collaborative_model.predict(uid=\3, iid=\4).est
\1    except (ValueError, TypeError) as e:
\1        # Nếu lỗi kiểu dữ liệu, thử với dạng chuỗi
\1        print(f"Converting IDs to string: {e}")
\1        \2 = collaborative_model.predict(uid=str(\3), iid=str(\4)).est
\1except Exception as e:
\1    print(f"Hybrid prediction failed for user {\3}, product {\4}: {e}")'''
    
    updated_content = re.sub(pattern2, replacement2, updated_content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    
    print(f"Updated {file_path}")

def update_model_predict_calls_in_directory(directory):
    """
    Updates model.predict calls in all Python files in the directory
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                update_model_predict_calls(file_path)

if __name__ == "__main__":
    # Update hybrid.py
    update_model_predict_calls('d:/python/recommendation-system/recommenders/hybrid.py')
    # Also update all other Python files in recommenders directory
    update_model_predict_calls_in_directory('d:/python/recommendation-system/recommenders')
