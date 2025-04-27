import openai
from config.config import Config
from database.mongodb import MongoDB

class ChatService:
    def __init__(self):
        openai.api_key = Config.OPENROUTER_API_KEY  # Key từ openrouter.ai
        openai.base_url = "https://openrouter.ai/api/v1"
        self.client = openai
        self.db = MongoDB()
        self.debug = True

    def log_debug(self, message, data=None):
        """Helper method để in debug logs"""
        if self.debug:
            print(f"\n{'='*50}")
            print(f"DEBUG: {message}")
            if data:
                print(f"DATA: {data}")
            print(f"{'='*50}\n")
        
    def extract_filters(self, user_message):
        self.log_debug("Processing user message:", user_message)

        """
        Sử dụng GPT để trích xuất thông tin từ câu hỏi
        """
        prompt = f"""
        Câu hỏi: {user_message}
        Hãy trích xuất:
        - Địa điểm (tỉnh/thành phố hoặc vùng)
        - Dịch vụ yêu cầu
        - Sức chứa tối thiểu (nếu có)
        Trả về JSON như: {{
          "province": "Đà Lạt",
          "services": ["xe đạp"],
          "min_capacity": 10
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            # self.log_debug("AI Response:", response)
            
            # Debug logs
            # print("Response type:", type(response))
            # print("Response content:", response)

            # Extract content from response
            if isinstance(response, dict):
                if 'choices' in response:
                    content = response['choices'][0]['message']['content']
                else:
                    content = response.get('content', '{}')
            elif hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = str(response)

            self.log_debug("Extracted content:", content)


            # Try to parse JSON from content
            import json
            try:
                filters = json.loads(content)
                self.log_debug("Parsed filters:", filters)

            except json.JSONDecodeError:
                self.log_debug("JSON parsing failed, trying eval")
                # If JSON parsing fails, try eval
                try:
                    filters = eval(content)
                    self.log_debug("Eval result:", filters)
                except:
                    print("Failed to parse response:", content)
                    return {"province": "", "services": [], "min_capacity": 0}

            # Validate and ensure required keys exist
            default_filters = {
                "province": "",
                "services": [],
                "min_capacity": 0
            }
            
            return {**default_filters, **filters}

        except Exception as e:
            print(f"Error in extract_filters: {str(e)}")
            return {"province": "", "services": [], "min_capacity": 0}
    def format_response(self, user_message, results):
        """
        Sử dụng GPT để format kết quả thành văn bản thân thiện
        """
        if not results:
            return "Xin lỗi, tôi không tìm thấy địa điểm nào phù hợp với yêu cầu của bạn."
            
        plain_text = "\n".join([
            f"{r['name']} - {r['address']} (Dịch vụ: {', '.join(r['services'])}, Sức chứa: {r['capacity']})" 
            for r in results
        ])
        
        prompt = f"""
        Người dùng hỏi: {user_message}
        Dưới đây là danh sách kết quả:
        {plain_text}
        Hãy trả lời ngắn gọn và thân thiện, có thể thêm emoji để làm cho tin nhắn sinh động hơn.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        # print("Response type:", type(response))
        # print("Response content:", response)

        return response.choices[0].message.content
    
    def process_query(self, user_message):
        try:
            self.log_debug("Starting query processing", user_message)
            
            filters = self.extract_filters(user_message)
            self.log_debug("Extracted filters:", filters)
            
            results = self.db.query_places(filters)
            self.log_debug("Database results:", results)
            
            reply = self.format_response(user_message, results)
            self.log_debug("Formatted response:", reply)
            
            return reply
        except Exception as e:
            self.log_debug("Error in process_query:", str(e))
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"