
class Cell:

    size = 0
    v = 0
    material = 0

    def __init__(self, size, material='air'):
        self.size = size
        self.v = size**3
        self.material = material
    
    def calculate_erosion(self, force):
        if self.material == 'air': pass
        else:
            pass

    def change_material(self,mat):
        self.material = mat

    