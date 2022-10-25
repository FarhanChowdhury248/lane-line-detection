## Preprocessing

- Gaussian Blur: Apply a gaussian blur to reduce noise. Note that the kernel size must be odd.
- Grayscale: We are not particularly concerned with colour, so grayscaling removes additional noise.

## Pipeline

- Canny Edge Detection: Get edges based on canny edge detection. Any edge with intensity gradient greater than `high_t` is guaranteed to be edge, and lower than `low_t` is guaranteed to be not an edge. Any edges in between must be connected to "sure-edge" pixels to be considered an edge.
- Masking: Make a reasonable assumption that lane lines will persist in the bottom half, and discard the top half with a mask, removing additional noise.
- Hough Lines Transform (HLT): Use the probabilistic hough line transform to get lines. `rho` and `theta` determine the degree of pixel and angles, and `threshold` determines the number of sinusoidal intersections required for a line to be detected. `min_line_len` and `max_line_gap` are self explanatory.
- Filter Lines: Filter the hough lines to keep ones whose absoloute slope falls in `[line_slope_min, line_slope_max]` and reaches 80% of the way to the bottom of the image.
- Cluster: A primitive clustering algorithm that groups based on slope only. To calculate the lane lines, the lines of each cluster are averaged. A final image is plotted for each test image, with all detected lines in green, and the calculated lane lines in red.

## Results

The algorithm performs well on lane lines with minimal curves and obstacles (cars), and can effectively filter out non-lane lines.

It struggles when there are many obstacles or sources of additional lines similar in structure to lane lines (shadows), which create additional lines detected by HLT. Additionally, since HLT detects straight continuous lines, curved and broken lane lines are poorly detected.

## Improvements

### Blocked

- Mask: A better mask can be implemented by utilizing the position and orientation of the camera with respect to the lanes.
- Distortion: If distortion matrix can be derived for camera, we can undistort input images for better image processing.
- Grass Lanes: No grass lanes datasets found online. We need to find some or generate our own.

### Future Versions

- Curves: Consider alternative approaches to detect curved lane lines. If camera properties are given, a perspective transform + lane line histogram + sliding window technique can be used. Until then, camera properties can be estimated provided consistent dataset.
- Removing Shadows + Unwanted lines: Consider adding alpha-based gradient amplification to highlight white lines and smoothen all other gradients.
