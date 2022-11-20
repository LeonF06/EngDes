# Create a class for the ball
class Ball:
    def __init__(self, center, radius, outline, area, colour, detected):
        self.center = center
        self.radius = radius
        self.outline = outline
        self.area = area
        self.colour = colour
        self.detected = detected

# Create a class for the blob
class Blob:
    def __init__(self, outline, area, detected):
        self.outline = outline
        self.area = area
        self.detected = detected

# Create a class for the defect
class Defect:
    def __init__(self, outline, area):
        self.outline = outline
        self.area = area