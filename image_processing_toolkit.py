"""
DSC 20 Mid-Quarter Project
Name(s): Krish Prasad
PID(s):  A17402508
"""

import numpy as np
from PIL import Image

NUM_CHANNELS = 3


# -------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# -------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    template for image objects
        """

    def __init__(self, pixels):
        """
        constructor for RGB image,
        assigns pixels and num_colums and num_rows

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        if not isinstance(pixels, list) or len(pixels) < 1:
            raise TypeError()
        for i in pixels:
            if not isinstance(i, list) or len(i) < 1:
                raise TypeError()
        temp = len(pixels[0])
        for i in pixels:
            if len(i) != temp:
                raise TypeError()
        for i in pixels:
            for j in i:
                if not isinstance(j, list) or len(j)!=3:
                    raise TypeError()
        
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        returns tuple of num rows and num cols


        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        returns deep copy of self.pixels

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[rgb for rgb in columns]for columns in rows] for rows\
        in self.pixels]
    
    def copy(self):
        """
        return copy of RGBImage
        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())
    

    def get_pixel(self, row, col):
        """

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if type(row) != int or type(col) != int:
            raise TypeError()
        if row>= self.size()[0] or col >= self.size()[1] or row < 0 or col<0:
            #print(self.size()[0], self.size()[1])
            raise ValueError()
        
        rgb_list = self.get_pixels()[row][col]
        return (rgb_list[0], rgb_list[1], rgb_list[2])


    def set_pixel(self, row, col, new_color):
        """
        sets a new rgb value to a pixel

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if type(row) != int or type(col) != int:
            raise TypeError()
        if row>= self.size()[0] or col >= self.size()[1] or row < 0 or col<0:
            raise ValueError()
        if type(new_color) != tuple or len(new_color)!= 3:
            raise TypeError()
        if not all([isinstance(i, int) for i in new_color]):
            raise TypeError()
        if not all([i<= 255 for i in new_color]):
            raise ValueError()

        for i in range(3):
            if new_color[i] >=0:
                self.pixels[row][col][i] = new_color[i]


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    implements several image processing methods
    """

    def __init__(self):
        """
        initializes instance variable cost to 0
        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        returns cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        negates each pixel value in an image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            #1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           #2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  #3
        >>> img_negate = img_proc.negate(img_input)                         #4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       #5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)#6
        """
        negate_image = image.get_pixels()
        negate_image = [[[(255-RGB) for RGB in columns] for columns in rows]\
        for rows in negate_image]
        return RGBImage(negate_image)
    
    def grayscale(self, image):
        """
        converts each pixel to grayscale

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        grayscale_image = image.get_pixels()
        grayscale_image = [[[(columns[0]+columns[1]+columns[2])//3,\
        (columns[0]+columns[1]+columns[2])//3,(columns[0]+columns[1]+\
        columns[2])//3] for columns in rows] for rows in grayscale_image]
        return RGBImage(grayscale_image)

    def rotate_180(self, image):
        """
        rotates the pixels in an image 180 degrees

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        rotate_180_images = image.get_pixels()
        rows = image.size()[0]
        columns = image.size()[1]
        rotate_180_images = [[rotate_180_images[columns-i_rows-1]\
        [rows-j_columns-1] for j_columns in range(columns)] for i_rows\
        in range(rows)]
        return RGBImage(rotate_180_images)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    monetized version of template class
    """

    def __init__(self):
        """
        initializes cost to 0

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.rotate_count=0
        self.coupon_count = 0  

    def negate(self, image):
        """
        adds +5 to the cost


        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)

        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        self.cost+=5
        return super().negate(image)
        

    def grayscale(self, image):
        """
        adds 6 to the cost
        """
        self.cost+=6
        return super().grayscale(image)
    
    def rotate_180(self, image):
        """
        adds 10 to the cost

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        self.rotate_count+=1
        if self.rotate_count ==2:
            self.cost-=10
            self.rotate_count=0
        elif self.coupon_count > 0:
            self.rotate_count+=1
        else:
            self.cost+=10

        return super().rotate_180(image)

    def redeem_coupon(self, amount):

        """
        adds an amount of coupons to the instance variable 
        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """
        self.coupon_count+= amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Premium verison of image processing app with extra features
    """

    def __init__(self):
        """
        initializes the initial cost to 50

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        changes all pixels with the specified color in the
        chroma_image to the pixels at the same places in the 
        background_image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        if type(chroma_image) != RGBImage or type(background_image)\
        != RGBImage:
            raise TypeError()
        if chroma_image.size() != background_image.size():
            raise ValueError()
        
        chroma = chroma_image.get_pixels()
        combined = background_image.get_pixels()
        for rows in range(len(combined)):
            for columns in range(len(combined[0])):
                if chroma[rows][columns] != list(color):
                    combined[rows][columns] = chroma[rows][columns]
        return RGBImage(combined)


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        add a sticker at a given x and y position to a background image

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        if type(sticker_image) != RGBImage or \
        type(background_image) != RGBImage:
            raise TypeError()
        if sticker_image.size()[0] > background_image.size()[1] \
        or sticker_image.size()[1] > background_image.size()[1]:
            raise ValueError()

        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()
        
        if x_pos + sticker_image.size()[0] >= background_image.size()[0] \
        or y_pos + sticker_image.size()[1] >= background_image.size()[1]:
            raise ValueError()

        background = background_image.get_pixels()
        sticker = sticker_image.get_pixels()
        for i in range(sticker_image.size()[0]):
            for j in range(sticker_image.size()[1]):
                background[x_pos+i][y_pos+j] = sticker[i][j]

        return RGBImage(background)
        
# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Implement K-nearest Neighbors classification
    
    # make random training data (type: List[Tuple[RGBImage, str]])
    >>> train = []

    # create training images with low intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
    ...     for _ in range(20)
    ... )

    # create training images with high intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
    ...     for _ in range(20)
    ... )

    # initialize and fit the classifier
    >>> knn = ImageKNNClassifier(5)
    >>> knn.fit(train)

    # should be "low"
    >>> print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
    low

    # can be either "low" or "high" randomly
    >>> print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
    This will randomly be either low or high

    # should be "high"
    >>> print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
    high
    """

    
    def __init__(self, n_neighbors):
        """
        init for imageknnclassifier
        """
        # YOUR CODE GOES HERE #
        self.n_neighbors = n_neighbors
        self.data = []

    def fit(self, data):
        """
        fits the classifier
        """
        if len(data) <= self.n_neighbors:
            raise ValueError()
        if len(self.data) >0:
            raise ValueError()
        
        self.data.extend(data)
        
    @staticmethod
    def distance(image1, image2):
        """
        calculates euclidian distance between two images
        """'''        for i in range(len(image1)):
            for j in range(len(image1[0])):
                for k in range(3):
                    total += (image1[i][j][k] - image2[i][j][k])**2'''"""
        """
        if type(image1)!=RGBImage or type(image2)!=RGBImage:
            raise TypeError()
        if image1.size() != image2.size():
            raise ValueError()

        img1 = image1.get_pixels()
        img2 = image2.get_pixels()

        total = sum([(img1[i][j][k] - img2[i][j][k])**2 for k in range(3) for j in range(len(img1[0])) for i in range(len(img1))])
        distance = total**0.5
        return distance

    @staticmethod
    def vote(candidates):
        """
        finds the most popular label
        """
        most_popular = {}
        for i in candidates:
            if i in most_popular:
                most_popular[i] +=1
            else:
                most_popular[i] = 1
        most_popular_candidate = sorted(most_popular, key=most_popular.get)[-1]
        return most_popular_candidate

    def predict(self, image):
        """
        
        """
        if len(self.data) == 0:
            raise ValueError()

        distance = [(i[1], self.distance(image, i[0])) for i in self.data]
        distance = sorted(distance, key=lambda x:x[1])
        low_or_high = [i[0] for i in distance[:5]]
        return self.vote(low_or_high)

def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    """
    return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()
