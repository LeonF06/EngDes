# Create a class for the ball
'''class Ball:
    def __init__(self, outline, area, colour, detected):
        self.outline = outline
        self.area = area
        self.colour = colour
        self.detected = detected'''

class Ball:
    def __init__(self, center, radius, area, colour, detected):
        self.center = center
        self.radius = radius
        self.area = area
        self.colour = colour
        self.detected = detected

# Create a class for inner circle
class InnerCircle:
    def __init__(self, outline, area, detected):
        self.outline = outline
        self.area = area
        self.detected = detected

# Create a class for small cirle
class SmallCircle:
    def __init__(self, outline, area, detected):
        self.outline = outline
        self.area = area
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

# Create a class for the filler hole
class FillerHole:
    def __init__(self, outline, area, detected):
        self.outline = outline
        self.area = area
        self.detected = detected