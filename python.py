import cv2
import numpy as np

def grayscale(image): return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def gaussian_blur(image, kernel_size): return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
def canny(image, low_threshold, high_threshold): return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)

def get_birds_eye_view(image):
    h, w = image.shape[:2]
    src = np.float32([
        [200, h],            
        [w - 200, h],        
        [w // 2 + 70, int(h * 0.62)], 
        [w // 2 - 70, int(h * 0.62)]  
    ])
    dst = np.float32([
        [300, h],
        [w - 300, h],
        [w - 300, 0],
        [300, 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, Minv

def fit_curved_lanes(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 10
    window_height = int(binary_warped.shape[0] // nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 60
    minpix = 30

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix: leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix: rightx_current = int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        return None, None

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds] 
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    if len(leftx) > 50 and len(rightx) > 50:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit
    return None, None

def draw_curved_lanes(image, warped, left_fit, right_fit, Minv):
    if left_fit is None or right_fit is None: return image
        
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    
    mask = np.zeros_like(newwarp)
    mask[int(image.shape[0]*0.62):, :] = 255
    newwarp = cv2.bitwise_and(newwarp, mask)
    
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)

def process_image(image):
    gray = grayscale(image)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 40, 100)
    
    h, w = image.shape[:2]
    vertices = np.array([[(50, h), (w//2-60, int(h*0.62)), (w//2+60, int(h*0.62)), (w-50, h)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)
    
    warped_roi, Minv = get_birds_eye_view(roi)
    left_fit, right_fit = fit_curved_lanes(warped_roi)
    result = draw_curved_lanes(image, warped_roi, left_fit, right_fit, Minv)
    return result

def main():
    video_path = 'video.mp4'  
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print("Error: Could not open video file.")
        return
    
    playback_speed = 0.5 
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30  
    delay = int((1.0 / playback_speed) * (1000.0 / fps))
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # This block automatically records and stores your video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_curved_lane.mp4', fourcc, fps, (frame_width, frame_height))
    
    print("Processing video and saving output...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            processed_frame = process_image(frame)
            out.write(processed_frame)  # Saves the frame to your disk
            
            cv2.imshow('Advanced Curved Lane Tracking', processed_frame)
            if cv2.waitKey(max(0.5, delay)) & 0xFF == ord('q'): break
        else: break
            
    cap.release()
    out.release()  # Closes the file safely
    cv2.destroyAllWindows()
    print("Success! Your output video is saved as 'output_curved_lane.mp4'.")

if __name__ == '__main__': main()
