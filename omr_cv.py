import time
import os
import cv2 as cv
import numpy as np
from scipy.stats import linregress
from shapely.geometry import LineString

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from PIL import Image


class OpticalMarkRecognitionCV:
    def __init__(self):
        """
        The __init__ function is the constructor for the class.
        
        :param self: Used to refer to the object itself.
        :return: None
        """
        self.inp_image = None
        self.bin_image = None
        self.bin_plus_img = None
        self.backdrop = None
        self.solved_img = None
        self.solved_compact_img = None
        self.marked_thresh = None
        self.valid_rects = []
        self.final_rows = []
        self.tf_model = self.load_tf_model()
    
    def start_omr_cv(self, img):
        """
        The start_omr_cv function takes in an image and starts the image evaluation process.
        
        :param self: Used to refer to the object itself.
        :param img: Used to pass the image to the function.
        :return: None
        """
        self.inp_image = None
        self.bin_image = None
        self.bin_plus_img = None
        self.backdrop = None
        self.valid_rects = []
        self.final_rows = []
        self.inp_image = img
        self.preprocess_image()
        self.box_finding_algorithm()
        self.find_remaining_boxes()

        self.sort_rows()
        # self.determine_checked_boxes()
        self.determine_checked_boxes_tf()
        # self.create_solution_image()

    def preprocess_image(self):
        """
        The preprocess_image function creates two binary images of the input image, 
        one of the binary images has an increased contrast applied to it.
        
        :param self: Used to access variables that belong to the class.
        :return: None
        """
        self.backdrop = np.zeros_like(self.inp_image, np.uint8)
        gauss_img = cv.GaussianBlur(self.inp_image, (3, 3), 0)
        gray = cv.cvtColor(gauss_img, cv.COLOR_BGR2GRAY)
        self.bin_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] 
        st_time = time.time()
        lab = cv.cvtColor(gauss_img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl = clahe.apply(l)
        limg = cv.merge((cl,a,b))
        color_contrast_image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
        gray_contrast_image = cv.cvtColor(color_contrast_image, cv.COLOR_BGR2GRAY)
        self.bin_plus_img = cv.threshold(gray_contrast_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    def box_finding_algorithm(self):
        """
        The box_finding_algorithm function is used to find the coordinates of the boxes in the solution sheet.
        
        :param self: Used to access the class attributes.
        :return: None
        """
        y_offset = 770
        x_offset = 130
        res = 1376
        found_rects = []
        croppped_bin = self.bin_image[y_offset: y_offset+res, x_offset: x_offset+res]
        
        contours = cv.findContours(croppped_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for countour in contours:
            peri = cv.arcLength(countour, True)
            approx = cv.approxPolyDP(countour, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv.boundingRect(approx)
                x, y = x + x_offset, y + y_offset
                if 25 < w < 33 and 25 < h < 33:
                    found_rects.append([x, y, w, h])
                    cv.rectangle(self.backdrop, (x, y), (x + w, y + h), (12, 36, 255), cv.FILLED)

        for rect in found_rects:
            self.valid_rects.append(rect)

    def find_remaining_boxes(self):
        """
        The find_remaining_boxes function finds the position of the remaining boxes 
        which were not found by the box_finding_algorithm.
        
        :param self: Used to refer to the object itself.
        :return: None
        """
        self.valid_rects.sort(key=lambda i: i[1], reverse=False)
        last_y = self.valid_rects[0][1]
        y_split_index = 0
        final_rows = []
        for index, rect in enumerate(self.valid_rects):
            if abs(rect[1] - last_y) > 10:
                final_rows.append(self.valid_rects[y_split_index: index])
                y_split_index = index
                last_y = self.valid_rects[index + 1][1]
            else:
                last_y = rect[1]
        final_rows.append(self.valid_rects[y_split_index:])

        for index, row in enumerate(final_rows):
            row.sort(key=lambda i: i[0], reverse=False)
            final_rows[index] = row
        
        self.valid_rects.sort(key=lambda i: i[0], reverse=False)
        last_x = self.valid_rects[0][0]
        x_split_index = 0
        final_columns = []
        for index, rect in enumerate(self.valid_rects):
            # print(rect[1] - last_y)
            if abs(rect[0] - last_x) > 10:
                final_columns.append(self.valid_rects[x_split_index: index])
                x_split_index = index
                last_x = self.valid_rects[index + 1][0]
            else:
                last_x = rect[0]
        final_columns.append(self.valid_rects[x_split_index:])

        for index, col in enumerate(final_columns):
            col.sort(key=lambda i: i[1], reverse=False)
            final_columns[index] = col
        
        x = np.arange(0, 1701)
        row_line_vals = []
        row_line_shapely_vals = []
        for row in final_rows:
            x_values = []
            y_values = []
            for rect in row:
                x_values.append(rect[0])
                y_values.append(rect[1])
            line_a, line_b, _r, _p, _std = linregress(x_values, y_values)
            row_line_vals.append(x*line_a + line_b)
            row_line_shapely_vals.append(LineString(np.column_stack((x, (x*line_a + line_b)))))

        x = np.arange(0, 2308)
        col_line_vals = []
        col_line_shapely_vals = []
        for col in final_columns:
            x_values = []
            y_values = []
            for rect in col:
                x_values.append(rect[1])
                y_values.append(rect[0])
            line_a, line_b, _r, _p, _std = linregress(x_values, y_values)

            col_line_vals.append(x*line_a + line_b)
            col_line_shapely_vals.append(LineString(np.column_stack(((x*line_a + line_b), x))))

        intersections = []

        for col_line in col_line_shapely_vals:
            for row_line in row_line_shapely_vals:
                intersections.append(col_line.intersection(row_line))

        self.valid_rects = []

        # for col_line_val in col_line_vals:
        #     plt.plot(col_line_val, x, "b")
        
        # x = np.arange(0, 1701)
        # for row_line_val in row_line_vals:
        #     plt.plot(x, row_line_val, "m")

        for intersection in intersections:
            x_pos, y_pos = intersection.xy
            self.valid_rects.append([round(x_pos[0]), round(y_pos[0]), 29, 29])
            # plt.plot(round(x_pos[0]), round(y_pos[0]), marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
        
        # plt.show()

    def sort_rows(self):
        """
        The sort_rows function first sorts the rectangles by their y-coordinate, splits them into rows, 
        and then sorts the order of the rectangles in each row by their x-coordinates.
        
        :param self: Used to refer to the object that is calling this function.
        :return: None
        """
        final_rows = []
        y_split_index = 0
        self.valid_rects.sort(key=lambda i: i[1], reverse=False)
        last_y = self.valid_rects[0][1]
        for index, rect in enumerate(self.valid_rects):
            if abs(rect[1] - last_y) > 10:
                final_rows.append(self.valid_rects[y_split_index: index])
                y_split_index = index
                last_y = self.valid_rects[index + 1][1]
            else:
                last_y = rect[1]
        final_rows.append(self.valid_rects[y_split_index:])

        for index, row in enumerate(final_rows):
            row.sort(key=lambda i: i[0], reverse=False)
            self.final_rows.append(row)

    def determine_checked_boxes(self):
        """
        The determine_checked_boxes function accomplishes two things:
        1. Gives each rectangle a value based on how "marked" they are.
        2. Finds the threshold to split unmarked boxes into marked boxes.
        
        :param self: Used to access the class attributes.
        :return: None
        """
        rect_values = []
        thresh_crop_size = 0
        for row_index, row in enumerate(self.final_rows):
            for rect_index, rect in enumerate(row):
                start_x, start_y, width, height = rect[0], rect[1], rect[2], rect[3]
                while (width < 34):
                    start_x -= 0.5
                    width += 1
                while (height < 34):
                    start_y -= 0.5
                    height += 1
                start_x, start_y = int(start_x), int(start_y)
                cropped_rect = self.bin_plus_img[start_y:start_y + height, start_x:start_x + width]
                value = cv.countNonZero(cropped_rect)
                self.final_rows[row_index][rect_index].append(value)
                rect_values.append(value)
        
        rect_values.sort()
        max_diff = 0
        max_index = 0
        index_offset = 115
        for index in range(len(rect_values) -1 - index_offset):
            diff = rect_values[index + 1 + index_offset] - rect_values[index + index_offset]
            if diff > max_diff:
                max_diff = diff
                max_index = index
        
        # plt.plot(rect_values)
        # plt.plot(max_index+index_offset, rect_values[max_index + index_offset], marker="o")
        # plt.show()

        self.marked_thresh = rect_values[max_index + index_offset]

    def return_marked_boxes_dict(self):
        """
        The return_marked_boxes_dict function specifically creates a dictionary of the marked boxes in the form:
        {Task 1: ['A', 'B', 'C'], Task 2: ['D', 'F']} and so on and returns this dicionary.
        
        :param self: Used to access the class attributes.
        :return: A dictionary with the marked boxes for each task.
        """
        character_arr = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        appendable_characters = []
        marked_sheet = {}

        for row_index, row in enumerate(self.final_rows):
            if row_index % 2 == 0:
                char_off = 0
            else:
                char_off = 13
            for rect_index, rect in enumerate(row):
                if rect[4] > self.marked_thresh:
                    appendable_characters.append(character_arr[rect_index+char_off])
                    cv.rectangle(self.backdrop, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)

            if row_index % 2 != 0:
                marked_sheet["Task "+str(int((row_index+1)/2))] = appendable_characters
                appendable_characters = []
        
        return marked_sheet
    
    @staticmethod
    def load_tf_model():
        """
        The load_tf_model function loads our keras model for our box predicitions.
        
        :return: The loaded keras model.
        """
        root_dir_path = os.path.dirname(__file__)
        model_dir_path = os.path.join(root_dir_path, "classification_model")
        return keras.models.load_model(model_dir_path)

    def determine_checked_boxes_tf(self):
        """
        The determine_checked_boxes_tf function loops through all of our rectangles and uses
        our keras model to predict if the rectangle is marked or not.
        
        :param self: Used to access the class variables.
        :return: None
        """
        self.marked_thresh = 0.5
        for row_index, row in enumerate(self.final_rows):
            for rect_index, rect in enumerate(row):
                val = self.predict_box(rect)
                self.final_rows[row_index][rect_index].append(val)

    def predict_box(self, rect):
        """
        The predict_box function takes in a bounding box and predicts if the 
        rectangle in the image is marked or not.
        
        :param self: Used to access the class variables.
        :param rect: Used to specify the location of the bounding box of the rectangle in the image.
        :return: bool: True if the box was marked and False if it is clear.
        """
        start_x, start_y, width, height = rect[0], rect[1], rect[2], rect[3]
        while (width < 32):
                start_x -= 0.5
                width += 1
        while (height < 32):
            start_y -= 0.5
            height += 1
        
        start_x = int(start_x)
        start_y = int(start_y)

        Y = keras.preprocessing.image.img_to_array(self.bin_plus_img[start_y: start_y+height, start_x: start_x+width])
        X = np.expand_dims(Y,axis=0)
        val = self.tf_model.predict(X)
        return val[0][0]
    
    def create_solution_image(self):
        """
        The create_solution_image function creates a two images where the marked and clear boxes are marked.
        The solved_compact_img is a compact version of the main solution image for better visibility.
        
        :param self: Used to access variables that belongs to the class.
        :return: None
        """
        solution_backdrop = np.zeros_like(self.inp_image, np.uint8)
        self.solved_img = self.inp_image.copy()
        for row_index, row in enumerate(self.final_rows):
            for rect_index, rect in enumerate(row):
                start_x, start_y, width, height = rect[0], rect[1], rect[2], rect[3]
                while (width < 40):
                    start_x -= 0.5
                    width += 1
                while (height < 40):
                    start_y -= 0.5
                    height += 1
                start_x, start_y = int(start_x), int(start_y)
                if rect[4] > self.marked_thresh:
                    cv.rectangle(solution_backdrop, (start_x, start_y), (start_x + width, start_y + height), (255, 0, 0), cv.FILLED)
                else:
                    cv.rectangle(solution_backdrop, (start_x, start_y), (start_x + width, start_y + height), (0, 0, 255), cv.FILLED)
        alpha = 0.5
        mask = solution_backdrop.astype(bool)
        self.solved_img[mask] = cv.addWeighted(self.inp_image, alpha, solution_backdrop, alpha, 0)[mask]

        # row_images = []
        # col_images = []
        # for row_index, row in enumerate(self.final_rows):
        #     col_images = []
        #     for rect_index, rect in enumerate(row):
        #         start_x, start_y, width, height = rect[0], rect[1], rect[2], rect[3]
        #         while (width < 34):
        #             start_x -= 0.5
        #             width += 1
        #         while (height < 34):
        #             start_y -= 0.5
        #             height += 1
        #         start_x, start_y = int(start_x), int(start_y)
        #         col_images.append(self.solved_img[start_y:start_y + height, start_x:start_x + width])
        #     row_images.append(np.concatenate((col_images[0], col_images[1], col_images[2], col_images[3], col_images[4], col_images[5], col_images[6], col_images[7], col_images[8], col_images[9], col_images[10], col_images[11], col_images[12]), axis=1))
        # self.solved_compact_img = np.concatenate((row_images[0], row_images[1], row_images[2], row_images[3], row_images[4], row_images[5], row_images[6], row_images[7], row_images[8], row_images[9], row_images[10], row_images[11], row_images[12], row_images[13], row_images[14], row_images[15], row_images[16], row_images[17], row_images[18], row_images[19]), axis=0)


if __name__ == "__main__":
    omr_cv = OpticalMarkRecognitionCV()
    image = cv.imread("modified_sheets/Sheet4_sp_LOW.jpg")
    omr_cv.start_omr_cv(image)
    
    image = cv.imread("modified_sheets/Sheet5_sp_LOW.jpg")
    omr_cv.start_omr_cv(image)
    