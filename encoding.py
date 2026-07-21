DIRECTIONS = {
    (0, 0): (1, 1, 1),
    (0, 1): (1, -1, -1),
    (1, 0): (-1, 1, -1),
    (1, 1): (-1, -1, 1),
}

OPPOSITE = {
    (0, 0): (0, 0),
    (0, 1): (0, 1),
    (1, 0): (1, 0),
    (1, 1): (1, 1),
}


def bits_to_directions(bitstring):
    return [(int(bitstring[2*i]), int(bitstring[2*i+1])) for i in range(len(bitstring) // 2)]
 
def bits_to_coords(bitstring):
    coords = [(0, 0, 0)]
    x, y, z = 0, 0, 0
    for (b0, b1) in bits_to_directions(bitstring):
    dx, dy, dz = DIRECTIONS[(b0, b1)]
    x, y, z = x + dx, y + dy, z + dz
    coords.append((x, y, z))
    return coords

