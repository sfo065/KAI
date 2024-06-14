class House:
    """
    Class for storing building information for a single building.
    Contains

    """
    def __init__(self, utm_coords,edges, id):
        self.utm_coords = utm_coords
        self.edges = edges
        self.utm_mean = np.mean(utm_coords, axis=1)
        self.image_ids = list()
        self.image_coords = list()
        self.id = id