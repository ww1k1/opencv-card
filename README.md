### Challenges

- Complex background with textures and logos
- Low contrast between digits and background
- Digits are grouped rather than isolated
- Noise and lighting variations

### Solutions

- TopHat operation to highlight bright digit regions
- Sobel operator to enhance vertical edges
- Morphological closing to connect digit groups
- Thresholding to simplify structure
- Contour filtering based on geometric constraints
- Template matching for final digit recognition
