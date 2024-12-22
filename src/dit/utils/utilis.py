# convert the image value [0, 1] to [-1, 1] to match the gaussian noise distribution
def apply_transform(func):
    def warpper(self, idx: int):
        image, idx = func(self, idx)
        image = image * 2 - 1 # convert to [-1, 1] to match the guassian noise distribution
        return image, idx
    return warpper