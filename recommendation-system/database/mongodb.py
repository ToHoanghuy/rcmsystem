from pymongo import MongoClient
from config.config import Config

class MongoDB:
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client.get_database()
        self.places = self.db['places']
    
    def query_places(self, filters):
        """
        Tìm kiếm địa điểm theo bộ lọc
        """
        mongo_filter = {
            "province": {"$regex": filters['province'], "$options": "i"},
            "services": {"$all": filters.get("services", [])},
            "capacity": {"$gte": filters.get("min_capacity", 0)}
        }
        return list(self.places.find(mongo_filter))

    def add_place(self, place_data):
        """
        Thêm địa điểm mới vào database
        """
        return self.places.insert_one(place_data)