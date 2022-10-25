## Preprocessing

- I investigated the test images manually to derive points for the perspective matrices. Check `perspective.py` for the results. This was used to approximate an average perspective matrix for the general case.
- No explicit blur was implemented as the sobel gradient and perspective projection introduced noise

## Pipeline

- Edge Detection: Canny Edge was modified to only consider the x gradient in the sobel operator. This would emphasize the importance of lane lines, which should be largely vertical w.r.t the camera.The sobel operator also only uses the light channel of the HLS value to base the gradient off of the contrast. The channel is also clipped with a minimum value of `150`, which achieves best results on the test set.
- Perspective Projection: An approximated projection transform was used to acquire a "top-down" view of the lanes.
- Cluster: A sliding window technique was used to extract two vertical(ish) lines from the projected image.
- Invert Projection: The projection was inverted on the lines to map them back to the original image.

## Results

The algorithm is more robust to changes in lighting due to the modified sobel gradient. The curve fitting also seeks to fit second order polynomials instead of lines, so curved lanes are better detected.

## Improvements

### Blocked

- Mask: A better mask can be implemented by utilizing the position and orientation of the camera with respect to the lanes.
- Distortion: If distortion matrix can be derived for camera, we can undistort input images for better image processing.
- Grass Lanes: No grass lanes datasets found online. We need to find some or generate our own.

### Future Versions

- Curves: Curve boundaries are imprecise. Consider using colour selection to filter out noisy edges.
- Perspective Projection: Projection matrix is unreliable and does not generalize well.
- We need a better way to filter out edges unrelated to lane lines.
