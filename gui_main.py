from importlib.metadata import PackageNotFoundError
import io
import socket
import os
import PySimpleGUI as sg
from PIL import Image
from omr_cv import OpticalMarkRecognitionCV
from excel_api import ExcelAPI
from metaclass import MetaClass
import cv2 as cv
import numpy as np
import threading
import copy

class GUIClass:
    def __init__(self, omr_cv: OpticalMarkRecognitionCV, excel_api: ExcelAPI, meta_data: MetaClass):
        self.omr_cv = omr_cv
        self.excel_api = excel_api
        self.meta_data = meta_data
        self.show_compact = False
        self.loaded_excel_file = False
        self.loaded_scans = False
        self.img_height = 920
        self.comp_rows = []

        self.row_size = (24, 1)
        sg.theme('DarkGrey10')

        mid_col = [[sg.Image(filename="", key='-MAIN IMAGE-')]]

        left_col = [[sg.Input(key='-LOAD EXCEL-', enable_events=True, visible=False)],
                  [sg.FileBrowse(button_text="Load Excel file", target='-LOAD EXCEL-', size=(29,2), file_types = (('Excel *.xlsx', '*.xlsx'),), font=("Helvetica",13))],
                  [sg.Text('No Excel file loaded', font="Helvetica 14", size=self.row_size, key="-EXCEL STATUS-", tooltip="", text_color="Red", justification="center")],

                  [sg.Text("")], 
                  [sg.Input(key='-LOAD SCANS-', enable_events=True, visible=False)],
                  [sg.FilesBrowse(button_text="Load scanned sheets", target='-LOAD SCANS-', size=(29,2), file_types = (('JPEG *.jpg', '*.jpg'),), font=("Arial",13))],
                  [sg.Text('No scans selected', font="Helvetica 14", size=self.row_size, key="-SCANS SELECTED STATUS-", tooltip="", text_color="Red", justification="center")],
                  [sg.Text("")],
                  [sg.Text("", font="Helvetica 14", size=self.row_size, key="-WARNING COUNT-", tooltip="", text_color="Red", justification="center")],
                  [sg.Text("")],
                  [sg.Text('', font="Helvetica 14", size=self.row_size, key="-CURRENT FILENAME-", tooltip="", justification="left")],
                  [sg.Text('', font="Helvetica 14", size=self.row_size, key="-SCANS PROGRESS STATUS-", tooltip="", justification="left")],
                  [sg.Button('Solve all Sheets', font="Helvetica 14", size=self.row_size, key="-SOLVE SHEETS-", tooltip="")],
        ]
        right_col = [
                    [sg.Image(filename="", key='-MAT IMAGE-')],
                    [sg.Text("Mat. number: ", font="Helvetica 14", size=self.row_size, key="-MAT NUM-", tooltip="", justification="left")],
                    [sg.Text('Name: ', font="Helvetica 14", size=self.row_size, key="-NAME-", tooltip="", justification="left")],
                    [sg.Text("")],
                    [sg.Button('Correct Sheet', font="Helvetica 14", size=self.row_size, key="-CORRECT SHEET-", tooltip="")],
                    [sg.Text("")],
                    [sg.Text("")],
                    [sg.Text("")],
                    [sg.Text('Sheet is unsaved', font="Helvetica 14", size=self.row_size, key="-CURRENT FILE SAVE STATUS-", tooltip="", justification="left", text_color="Red")],
                    [sg.Text("")],
                    [sg.Button('Save sheet in Excel', font="Helvetica 14", size=self.row_size, key="-SAVE IN EXCEL-", tooltip="")],
                    [sg.Text("")],
                    [sg.Button('Prev', font="Helvetica 14", size=(10,1), key="-PREV SHEET-", tooltip=""),
                    sg.Button('Next', font="Helvetica 14", size=(10,1), key="-NEXT SHEET-", tooltip="")],
        ]
        layout = [[sg.Column(left_col, vertical_alignment='top', element_justification="c"), sg.Column(mid_col, element_justification="c"),
                   sg.Column(right_col, vertical_alignment='top', element_justification="c")]]  # , sg.Column([io_col_1, sg.Text("asd", key="-IO EXPAND-"), io_col_2])

        
        self.window = sg.Window('Automatic Grading Software', layout, location=(55, 10), resizable=False, finalize=True)

        image = cv.imread("EmptySheet.jpg")
        mat_num_image = image[180:320, 1304: 1664]
        image = self.rescale_img(image, self.img_height)
        imgbytes = cv.imencode('.ppm', image)[1].tobytes()
        self.window['-MAIN IMAGE-'].update(data=imgbytes)

        mat_num_image = self.rescale_img(mat_num_image, 105)
        imgbytes = cv.imencode('.ppm', mat_num_image)[1].tobytes()
        self.window['-MAT IMAGE-'].update(data=imgbytes)
    
    def previous_sheet_callback(self):
        self.meta_data.decrease_index()
        self.update_gui_for_new_sheet()

    def next_sheet_callback(self):
        self.meta_data.increase_index()
        self.update_gui_for_new_sheet()
    
    def load_excel_callback(self, path):
        if self.excel_api.open_existing_file(path):
                    
            try:
                dir_path = os.path.dirname(path)
                new_dir = os.path.join(dir_path, "checked_images")
                self.meta_data.checked_images_path = new_dir
                os.mkdir(new_dir)
            except:
                pass
            self.update_excel_load_status()
            self.loaded_excel_file = True
        else:
            self.update_excel_load_status(False)
            self.loaded_excel_file = False
        self.check_scans_excel_status()
    
    def load_scans_callback(self, scans_str):
        scans_list = scans_str.split(";")
        if scans_str == "":
            self.update_scans_selected_status(False)
            self.loaded_scans = False
        else:
            self.meta_data.add_new_sheets(scans_list)
            self.update_scans_selected_status()
            self.update_gui_for_new_sheet()
            self.loaded_scans = True
        self.check_scans_excel_status()
    
    def combine_meta_dict(self):
        if len(self.excel_api.name_list) != len(self.meta_data.scans_list):
            self.window["-WARNING COUNT-"].update(f"Warning: number mismatch")
        else:
            self.window["-WARNING COUNT-"].update("")
        max_index = min(len(self.excel_api.name_list), len(self.meta_data.scans_list))
        for index in range(max_index):
            self.meta_data.meta_class_dict[self.meta_data.scans_list[index]]["mat_num"] = self.excel_api.name_list[index][0]
            self.meta_data.meta_class_dict[self.meta_data.scans_list[index]]["name"] = self.excel_api.name_list[index][1]
            self.meta_data.meta_class_dict[self.meta_data.scans_list[index]]["examtype"] = self.excel_api.name_list[index][2]

    def solve_sheets_callback(self):
        self.window["-PREV SHEET-"].update(disabled=True)
        self.window["-NEXT SHEET-"].update(disabled=True)
        self.window["-SOLVE SHEETS-"].update(disabled=True)
        for index, sheet in enumerate(self.meta_data.scans_list):
            self.meta_data.scan_index = index
            image = cv.imread(self.meta_data.scans_list[index])
            self.omr_cv.start_omr_cv(image)
            self.meta_data.meta_class_dict[self.meta_data.scans_list[index]]["marked_dict"] = self.omr_cv.return_marked_boxes_dict()
            self.meta_data.meta_class_dict[self.meta_data.scans_list[index]]["final_rows"] = self.omr_cv.final_rows
            self.meta_data.meta_class_dict[self.meta_data.scans_list[index]]["threshold"] = self.omr_cv.marked_thresh
            self.update_gui_for_new_sheet()
        self.meta_data.scan_index = 0
        self.update_gui_for_new_sheet()
        self.window["-SAVE IN EXCEL-"].update(disabled=False)
        self.window["-CORRECT SHEET-"].update(disabled=False)
        self.window["-PREV SHEET-"].update(disabled=False)
        self.window["-NEXT SHEET-"].update(disabled=False)
    
    def correct_box_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            for row_index, row in enumerate(self.comp_rows):
                for rect_index, rect in enumerate(row):
                    if ((rect[0] < x < rect[0] + rect[2]) and (rect[1] < y < rect[1] + rect[3])):
                        if rect[4] < self.omr_cv.marked_thresh:
                            value = self.omr_cv.marked_thresh + 0.5
                        else:
                            value = self.omr_cv.marked_thresh - 0.5

                        buf_rect = self.comp_rows[row_index][rect_index]
                        buf_rect[4] = value
                        self.comp_rows[row_index][rect_index] = buf_rect

                        buf_rect = self.omr_cv.final_rows[row_index][rect_index]
                        buf_rect[4] = value
                        self.omr_cv.final_rows[row_index][rect_index] = buf_rect
                        self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["marked_dict"] = self.omr_cv.return_marked_boxes_dict()
                        self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["final_rows"] = self.omr_cv.final_rows
                        self.create_solved_img()
    
    def correct_image_callback(self):
        scale_factor = self.img_height/omr_cv.solved_img.shape[0]
        self.comp_rows = copy.deepcopy(omr_cv.final_rows)
        for row_index, row in enumerate(self.comp_rows):
            for rect_index, rect in enumerate(row):
                start_x, start_y, width, height = rect[0], rect[1], rect[2], rect[3]
                while (width < 40):
                    start_x -= 0.5
                    width += 1
                while (height < 40):
                    start_y -= 0.5
                    height += 1
                self.comp_rows[row_index][rect_index] = [int(start_x*scale_factor), int(start_y*scale_factor), int(width*scale_factor), int(height*scale_factor), rect[4]]

        cv.namedWindow("Sheet")
        cv.setMouseCallback("Sheet", self.correct_box_callback)
        while cv.getWindowProperty("Sheet", cv.WND_PROP_VISIBLE) >= 1:
            cv.imshow("Sheet", self.rescale_img(omr_cv.solved_img, self.img_height))
            key = cv.waitKey(1) & 0xFF
        self.update_gui_for_new_sheet()
    
    def save_in_excel_callback(self):
        path, file_ = os.path.split(self.meta_data.scans_list[self.meta_data.scan_index])
        self.excel_api.save_in_sheet(self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["mat_num"], self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["name"], self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["marked_dict"], file_, self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["examtype"])
        self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["saved"] = True
        cv.imwrite(self.meta_data.checked_images_path + "/" + file_, omr_cv.solved_img)
        self.update_save_info()

    def check_scans_excel_status(self):
        if self.loaded_excel_file and self.loaded_scans:
            disable = False
            self.combine_meta_dict()
            self.update_gui_for_new_sheet()
        else:
            disable = True
        self.window["-PREV SHEET-"].update(disabled=disable)
        self.window["-NEXT SHEET-"].update(disabled=disable)
        self.window["-SOLVE SHEETS-"].update(disabled=disable)
        self.window["-CORRECT SHEET-"].update(disabled=True)
        self.window["-SAVE IN EXCEL-"].update(disabled=True)

    def update_excel_load_status(self, success=True):
        if success:
            self.window["-EXCEL STATUS-"].update("Excel file loaded", text_color='green')
        else:
            self.window["-EXCEL STATUS-"].update("Error loading/creating excel file", text_color='red')
    
    def update_scans_selected_status(self, success=True):
        if success:
            self.window["-SCANS SELECTED STATUS-"].update(str(len(self.meta_data.scans_list)) + " scans loaded", text_color='green')
        else:
            self.window["-SCANS SELECTED STATUS-"].update("Error loading scans", text_color='red')

    def update_scans_progress_status(self):
        self.window["-SCANS PROGRESS STATUS-"].update("Sheet " + str(self.meta_data.scan_index+1) + " of " + str(len(self.meta_data.scans_list)))
    
    def update_gui_for_new_sheet(self):
        self.show_compact = False
        self.update_scans_progress_status()
        self.update_filename()
        if self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["final_rows"] is not None:
            image = self.create_solved_img()
        else:
            image = cv.imread(self.meta_data.scans_list[self.meta_data.scan_index])
        mat_num_image = image[180:320, 1304: 1664]
        image = self.rescale_img(image, self.img_height)
        imgbytes = cv.imencode('.ppm', image)[1].tobytes()
        self.window['-MAIN IMAGE-'].update(data=imgbytes)

        mat_num_image = self.rescale_img(mat_num_image, 105)
        imgbytes = cv.imencode('.ppm', mat_num_image)[1].tobytes()
        self.window['-MAT IMAGE-'].update(data=imgbytes)
        self.update_meta_infomation()
        self.update_save_info()
    
    def update_filename(self):
        path, file = os.path.split(self.meta_data.scans_list[self.meta_data.scan_index])
        self.window['-CURRENT FILENAME-'].update(file)
    
    def create_solved_img(self):
        omr_cv.inp_image = cv.imread(self.meta_data.scans_list[self.meta_data.scan_index])
        omr_cv.final_rows = self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["final_rows"]
        omr_cv.marked_thresh = self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["threshold"]
        omr_cv.create_solution_image()
        return omr_cv.solved_img
        

    # def update_excel_meta_text(self):
    #     if self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["saved"]:
    #         name, score, mat_number = self.excel_api.fetch_meta_data(self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["mat_num"])
    #         self.window["-CURRENT FILE SAVE STATUS-"].update("Sheet is saved", text_color='green')
    #         self.window["-CURRENT FILE SAVED NUMBER-"].update("Number: " + str(mat_number))
    #         self.window["-CURRENT FILE SAVED NAME-"].update("Name: " + str(name))
    #         self.window["-CURRENT FILE SAVED SCORE-"].update("Score: " + str(round(score*100))+"%")
    #     else:
    #         self.window["-CURRENT FILE SAVE STATUS-"].update("Sheet is unsaved", text_color='red')
    #         self.window["-CURRENT FILE SAVED NUMBER-"].update("")
    #         self.window["-CURRENT FILE SAVED NAME-"].update("")
    #         self.window["-CURRENT FILE SAVED SCORE-"].update("")
    def update_meta_infomation(self):
        mat_num = str(self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["mat_num"])
        name = self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["name"]
        self.window["-MAT NUM-"].update("Mat. number: " + str(mat_num))
        self.window["-NAME-"].update("Name: " + str(name))
    
    def update_save_info(self):
        if self.meta_data.meta_class_dict[self.meta_data.scans_list[self.meta_data.scan_index]]["saved"]:
            self.window["-CURRENT FILE SAVE STATUS-"].update("Sheet is saved", text_color="Green")
        else:
            self.window["-CURRENT FILE SAVE STATUS-"].update("Sheet is unsaved", text_color="Red")


    def rescale_img(self, img, height):
        scale = height / img.shape[0]
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        return cv.resize(img, dim, interpolation=cv.INTER_AREA)


    def main(self):
        self.check_scans_excel_status()
        while True:
            event, values = self.window.Read(timeout=33)
            if event in ['Exit', sg.WIN_CLOSED]:
                break
            elif event == "-PREV SHEET-":
                self.previous_sheet_callback()
            
            elif event == "-NEXT SHEET-":
                self.next_sheet_callback()
            
            elif event == "-LOAD EXCEL-":
                status = self.load_excel_callback(values["-LOAD EXCEL-"])
            
            elif event == "-LOAD SCANS-":
                self.load_scans_callback(values["-LOAD SCANS-"])
            
            elif event == "-NEXT SHEET-":
                self.next_sheet_callback()

            elif event == "-SOLVE SHEETS-":
                threading.Thread(target=self.solve_sheets_callback, daemon=True).start()
            
            elif event == "-CORRECT SHEET-":
                self.correct_image_callback()
            
            elif event == "-SAVE IN EXCEL-":
                self.save_in_excel_callback()

if __name__ == "__main__":
    omr_cv = OpticalMarkRecognitionCV()
    excel_api = ExcelAPI()
    meta_data = MetaClass()
    gui = GUIClass(omr_cv, excel_api, meta_data)
    gui.main()