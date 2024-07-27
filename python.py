import cv2
import numpy as np

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines, color=[0, 255, 0], thickness=2):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if len(line) == 1:
                line = line[0]
            if len(line) == 4:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return combined_image

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def separate_lines(lines, img_shape):
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))
    return left_lines, right_lines

def average_slope_intercept(lines):
    if len(lines) == 0:
        return None
    slope, intercept = np.mean(lines, axis=0)
    return slope, intercept

def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if abs(slope) < 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [(x1, y1), (x2, y2)]

def process_image(image):
    gray = grayscale(image)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 50, 150)
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] / 2 - 50, imshape[0] / 2 + 50), 
                          (imshape[1] / 2 + 50, imshape[0] / 2 + 50), (imshape[1], imshape[0])]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)
    lines = hough_lines(roi, 1, np.pi / 180, 15, 40, 20)
    if lines is None:
        return image
    left_lines, right_lines = separate_lines(lines, imshape)
    left_lane = average_slope_intercept(left_lines)
    right_lane = average_slope_intercept(right_lines)
    y1 = imshape[0]
    y2 = int(y1 * 0.6)
    left_line_points = make_line_points(y1, y2, left_lane)
    right_line_points = make_line_points(y1, y2, right_lane)

    line_image = np.zeros_like(image)
    if left_line_points is not None:
        cv2.line(line_image, left_line_points[0], left_line_points[1], [0, 255, 0], 5)
    if right_line_points is not None:
        cv2.line(line_image, right_line_points[0], right_line_points[1], [0, 255, 0], 5)
    
    
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return result

def main():
    video_path = 'video.mp4'  
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.") 
        return
    
    
    playback_speed = 0.5
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            processed_frame = process_image(frame)
            cv2.imshow('Lane Detection', processed_frame)
            
            if cv2.waitKey(int(1 / playback_speed * 1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
