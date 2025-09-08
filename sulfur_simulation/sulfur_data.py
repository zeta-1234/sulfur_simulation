from __future__ import annotations

import numpy as np

DEFECT_LOCATIONS = np.array(
    [
        [40, 15],
        [40, 16],
        [40, 17],
        [40, 18],
        [41, 19],
        [41, 20],
        [41, 21],
        [41, 22],
        [42, 23],
        [42, 24],
        [42, 25],
        [42, 26],
        [43, 27],
        [43, 28],
        [43, 29],
        [43, 30],
        [44, 31],
        [44, 32],
        [44, 33],
        [44, 34],
        [45, 35],
        [45, 36],
        [45, 37],
        [45, 38],
        [46, 39],
        [46, 40],
        [46, 41],
        [46, 42],
        [47, 43],
        [47, 44],
        [47, 45],
        [47, 46],
        [48, 47],
        [48, 48],
        [48, 49],
        [48, 50],
        [49, 51],
        [49, 52],
        [49, 53],
        [49, 54],
        [50, 55],
        [50, 56],
        [50, 57],
        [50, 58],
        [51, 59],
        [51, 60],
        [51, 61],
        [51, 62],
        [52, 63],
        [52, 64],
        [52, 65],
        [52, 66],
        [53, 67],
        [53, 68],
        [53, 69],
        [53, 70],
        [54, 71],
        [54, 72],
        [54, 73],
        [54, 74],
        [55, 75],
        [55, 76],
        [55, 77],
        [55, 78],
        [56, 79],
        [56, 80],
        [56, 81],
        [56, 82],
        [57, 83],
        [57, 84],
        [57, 85],
    ],
    dtype=np.int16,
)

#  the following table lists which hcp sites are blocked when a layer is filled by atoms
#  could also be used to decide which atoms can hop into the layer if empty
#  the sites are listed as relative to an origin site for one repeating cell
blocked_hcp_sites = {
    0: {
        0: np.array([np.array([0, 0]), np.array([0, -1]), np.array([1, -1])]),
        1: np.array(
            [
                np.array([2, -2]),
                np.array([3, -2]),
                np.array([3, -3]),
                np.array([4, -3]),
            ]
        ),
        2: np.array([np.array([5, -3]), np.array([5, -4]), np.array([6, -4])]),
        3: np.array(
            [
                np.array([7, -5]),
                np.array([7, -6]),
                np.array([8, -5]),
                np.array([8, -6]),
            ]
        ),
        4: np.array([np.array([2, -3]), np.array([2, -4]), np.array([3, -4])]),
        5: np.array(
            [
                np.array([4, -5]),
                np.array([5, -5]),
                np.array([5, -6]),
                np.array([6, -6]),
            ]
        ),
        6: np.array([np.array([7, -7]), np.array([8, -7]), np.array([8, -8])]),
        7: np.array(
            [
                np.array([2, -5]),
                np.array([2, -6]),
                np.array([3, -5]),
                np.array([3, -6]),
            ]
        ),
        8: np.array([np.array([4, -6]), np.array([4, -7]), np.array([5, -7])]),
        9: np.array(
            [
                np.array([6, -8]),
                np.array([7, -8]),
                np.array([7, -9]),
                np.array([8, -9]),
            ]
        ),
    },
    1: {
        0: np.array([np.array([0, 0]), np.array([0, 1]), np.array([1, 0])]),
        1: np.array(
            [
                np.array([2, 0]),
                np.array([3, -1]),
                np.array([3, 0]),
                np.array([4, -1]),
            ]
        ),
        2: np.array([np.array([5, -1]), np.array([5, -2]), np.array([6, -2])]),
        3: np.array(
            [
                np.array([7, -1]),
                np.array([7, -2]),
                np.array([8, -2]),
                np.array([8, -3]),
            ]
        ),
        4: np.array([np.array([2, 1]), np.array([2, 2]), np.array([3, 1])]),
        5: np.array(
            [
                np.array([4, 1]),
                np.array([5, 1]),
                np.array([5, 0]),
                np.array([6, 0]),
            ]
        ),
        6: np.array([np.array([7, 0]), np.array([8, 0]), np.array([8, -1])]),
        7: np.array(
            [
                np.array([2, 4]),
                np.array([2, 3]),
                np.array([3, 3]),
                np.array([3, 2]),
            ]
        ),
        8: np.array([np.array([4, 3]), np.array([4, 2]), np.array([5, 2])]),
        9: np.array(
            [
                np.array([6, 2]),
                np.array([7, 2]),
                np.array([7, 1]),
                np.array([8, 1]),
            ]
        ),
    },
    2: {
        0: np.array([np.array([0, 0]), np.array([0, 1]), np.array([1, 0])]),
        1: np.array(
            [
                np.array([-1, 3]),
                np.array([-1, 4]),
                np.array([0, 2]),
                np.array([0, 3]),
            ]
        ),
        2: np.array([np.array([-2, 5]), np.array([-2, 6]), np.array([-1, 5])]),
        3: np.array(
            [
                np.array([-3, 8]),
                np.array([-2, 7]),
                np.array([-2, 8]),
                np.array([-1, 7]),
            ]
        ),
        4: np.array([np.array([1, 2]), np.array([1, 3]), np.array([2, 2])]),
        5: np.array(
            [
                np.array([0, 5]),
                np.array([0, 6]),
                np.array([1, 4]),
                np.array([1, 5]),
            ]
        ),
        6: np.array([np.array([-1, 8]), np.array([0, 7]), np.array([0, 8])]),
        7: np.array(
            [
                np.array([2, 3]),
                np.array([3, 2]),
                np.array([3, 3]),
                np.array([4, 2]),
            ]
        ),
        8: np.array([np.array([2, 4]), np.array([2, 5]), np.array([3, 4])]),
        9: np.array(
            [
                np.array([1, 7]),
                np.array([1, 8]),
                np.array([2, 6]),
                np.array([2, 7]),
            ]
        ),
    },
    3: {
        0: np.array([np.array([-1, 0]), np.array([-1, 1]), np.array([0, 0])]),
        1: np.array(
            [
                np.array([-3, 3]),
                np.array([-3, 4]),
                np.array([-2, 2]),
                np.array([-2, 3]),
            ]
        ),
        2: np.array([np.array([-4, 5]), np.array([-4, 6]), np.array([-3, 5])]),
        3: np.array(
            [
                np.array([-6, 7]),
                np.array([-6, 8]),
                np.array([-5, 7]),
                np.array([-5, 8]),
            ]
        ),
        4: np.array([np.array([-4, 2]), np.array([-4, 3]), np.array([-3, 2])]),
        5: np.array(
            [
                np.array([-6, 5]),
                np.array([-6, 6]),
                np.array([-5, 4]),
                np.array([-5, 5]),
            ]
        ),
        6: np.array([np.array([-8, 8]), np.array([-7, 7]), np.array([-7, 8])]),
        7: np.array(
            [
                np.array([-6, 2]),
                np.array([-6, 3]),
                np.array([-5, 2]),
                np.array([-5, 3]),
            ]
        ),
        8: np.array([np.array([-7, 4]), np.array([-7, 5]), np.array([-6, 4])]),
        9: np.array(
            [
                np.array([-9, 7]),
                np.array([-9, 8]),
                np.array([-8, 6]),
                np.array([-8, 7]),
            ]
        ),
    },
    4: {
        0: np.array([np.array([-1, 0]), np.array([-1, 1]), np.array([0, 0])]),
        1: np.array(
            [
                np.array([-3, -1]),
                np.array([-3, 0]),
                np.array([-2, -1]),
                np.array([-2, 0]),
            ]
        ),
        2: np.array([np.array([-4, -2]), np.array([-4, -1]), np.array([-3, -2])]),
        3: np.array(
            [
                np.array([-6, -2]),
                np.array([-6, -1]),
                np.array([-5, -3]),
                np.array([-5, -2]),
            ]
        ),
        4: np.array([np.array([-4, 1]), np.array([-4, 2]), np.array([-3, 1])]),
        5: np.array(
            [
                np.array([-6, 0]),
                np.array([-6, 1]),
                np.array([-5, 0]),
                np.array([-5, 1]),
            ]
        ),
        6: np.array([np.array([-8, 0]), np.array([-7, -1]), np.array([-7, 0])]),
        7: np.array(
            [
                np.array([-6, 3]),
                np.array([-6, 4]),
                np.array([-5, 2]),
                np.array([-5, 3]),
            ]
        ),
        8: np.array([np.array([-7, 2]), np.array([-7, 3]), np.array([-6, 2])]),
        9: np.array(
            [
                np.array([-9, 1]),
                np.array([-9, 2]),
                np.array([-8, 1]),
                np.array([-8, 2]),
            ]
        ),
    },
    5: {
        0: np.array([np.array([0, -1]), np.array([0, 0]), np.array([1, -1])]),
        1: np.array(
            [
                np.array([-1, -3]),
                np.array([-1, -2]),
                np.array([0, -3]),
                np.array([0, -2]),
            ]
        ),
        2: np.array([np.array([-2, -4]), np.array([-2, -3]), np.array([-1, -4])]),
        3: np.array(
            [
                np.array([-3, -5]),
                np.array([-2, -6]),
                np.array([-2, -5]),
                np.array([-1, -6]),
            ]
        ),
        4: np.array([np.array([1, -4]), np.array([1, -3]), np.array([2, -4])]),
        5: np.array(
            [
                np.array([0, -6]),
                np.array([0, -5]),
                np.array([1, -6]),
                np.array([1, -5]),
            ]
        ),
        6: np.array([np.array([-1, -7]), np.array([0, -8]), np.array([0, -7])]),
        7: np.array(
            [
                np.array([2, -5]),
                np.array([3, -6]),
                np.array([3, -5]),
                np.array([4, -6]),
            ]
        ),
        8: np.array([np.array([2, -7]), np.array([2, -6]), np.array([3, -7])]),
        9: np.array(
            [
                np.array([1, -9]),
                np.array([1, -8]),
                np.array([2, -9]),
                np.array([2, -8]),
            ]
        ),
    },
}

#  dictionaries of vectors for moving between tiles in different orientations
hcp_horizontal_vector = {
    0: np.array([2, -7]),
    1: np.array([2, 5]),
    2: np.array([5, 2]),
    3: np.array([-7, 2]),
    4: np.array([-7, 5]),
    5: np.array([5, -7]),
}
hcp_vertical_vector = {
    0: np.array([7, -2]),
    1: np.array([7, -5]),
    2: np.array([-5, 7]),
    3: np.array([-2, 7]),
    4: np.array([-2, -5]),
    5: np.array([-5, -2]),
}

#  relative positions of each sulfur in one tile#
relative_tile_positions = {
    0: np.array([0, 0]),
    1: np.array([1, 0]),
    2: np.array([2, 0]),
    3: np.array([3, 0]),
    4: np.array([1, -1]),
    5: np.array([2, -1]),
    6: np.array([3, -1]),
    7: np.array([1, -2]),
    8: np.array([2, -2]),
    9: np.array([3, -2]),
}
