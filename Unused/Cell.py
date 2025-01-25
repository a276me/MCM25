


class Cell:

    size = 0
    v = 0
    material = 0
    coord = 0

    def __init__(self, size, coord, material='air'):
        self.size = size
        self.v = size**3
        self.material = material
        self.coord = coord
    
    def calculate_erosion(self, force):
        if self.material == 'air': pass
        else:
            self.v -= force
        
            if self.v <= 0 and self.material == 'stairs':
                self.v = 0
                self.material = 'air'

    def change_material(self,mat):
        self.material = mat

    