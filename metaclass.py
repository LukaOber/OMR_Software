

class MetaClass:
    def __init__(self):
        self.checked_images_path = ""
        self.meta_class_dict = {}
        self.scans_list = []
        self.scan_index = 0
    
    def add_new_sheets(self, sheet_list):
        self.scans_list = sheet_list
        for sheet_path in sheet_list:
            self.meta_class_dict[sheet_path] = {
                "saved": False,
                "mat_num": None,
                "name": None,
                "examtype": None,
                "final_rows": None,
                "threshold": None,
                "marked_dict": None
            }
    
    def update_mat_num(self, sheet_path, mat_num):
        self.meta_class_dict[sheet_path] = {
            "mat_num": mat_num
        }

    def update_marked_dict(self, sheet_path, marked_dict):
        self.meta_class_dict[sheet_path] = {
            "marked_dict": marked_dict 
        }
    
    def increase_index(self):
        self.scan_index += 1
        if self.scan_index >= len(self.scans_list):
            self.scan_index = 0
    
    def decrease_index(self):
        self.scan_index -= 1
        if self.scan_index < 0:
            self.scan_index = len(self.scans_list) - 1