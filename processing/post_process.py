import cv2
import numpy as np
from typing import List, Tuple


def find_longest_edge(edge_lst: List[np.ndarray]) -> np.ndarray:
    len_edge_lst = [np.linalg.norm((edge_lst[i][2] - edge_lst[i][0], edge_lst[i][3] - edge_lst[i][1]))
                    for i in range(len(edge_lst))]
    max_idx = np.argmax(len_edge_lst)
    return edge_lst[max_idx]


def angle_with_horizontal_axis(checked_line: np.ndarray) -> float:
    a1 = checked_line[2] - checked_line[0]
    b1 = checked_line[3] - checked_line[1]
    cos_phi = np.abs(a1) / np.sqrt(a1 ** 2 + b1 ** 2)
    angle = float(np.arccos(cos_phi))
    return angle


def intersection_of_2lines(line1: np.ndarray, line2: np.ndarray) -> Tuple[int, int]:
    a1 = line1[0] - line1[2]
    b1 = line1[1] - line1[3]
    a2 = line2[0] - line2[2]
    b2 = line2[1] - line2[3]
    factor_matrix = [[b1, b2],
                     [-a1, -a2],
                     [b1 * line1[0] - a1 * line1[1], b2 * line2[0] - a2 * line2[1]]]

    factor_matrix_d = np.linalg.det(factor_matrix[:2])
    factor_matrix_dx = np.linalg.det([factor_matrix[2], factor_matrix[0]])
    factor_matrix_dy = np.linalg.det([factor_matrix[1], factor_matrix[2]])
    x = factor_matrix_dx / factor_matrix_d
    y = factor_matrix_dy / factor_matrix_d

    return int(-y), int(-x)


def line_center(line: np.ndarray) -> Tuple[int, int]:
    return (line[0] + line[2]) // 2, (line[1] + line[3]) // 2


def cluster_lines(lines: List[np.ndarray],
                  vector2center: np.ndarray,
                  num_cluster: int = 2,
                  horizontal: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    _, labels, centers = cv2.kmeans(vector2center[:, horizontal].reshape(-1, 1).astype(np.float32),
                                    K=num_cluster, bestLabels=None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.0001),
                                    attempts=10, flags=cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    cluster_edges_1 = [lines[a] for a in np.where(labels == 1)[0]]
    cluster_edges_2 = [lines[a] for a in np.where(labels == 0)[0]]

    return cluster_edges_1, cluster_edges_2


def get_max_len_edges(lines: List[np.ndarray],
                      line2center_vector: List[Tuple[int, int]],
                      horizontal: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    if len(lines) == 1:
        edges_1 = lines
        edges_2 = lines
    else:
        line2center_vector_np = np.reshape(np.array(line2center_vector, dtype=np.int32), (-1, 2))
        edges_1, edges_2 = cluster_lines(lines, line2center_vector_np, horizontal=horizontal)

    max_len_edge_1 = find_longest_edge(edges_1) if edges_1 else find_longest_edge(lines)
    max_len_edge_2 = find_longest_edge(edges_2) if edges_2 else find_longest_edge(lines)

    return max_len_edge_1, max_len_edge_2


def find_missed_edge(max_len_edge_1: np.ndarray, max_len_edge_2: np.ndarray,
                     center_x: int, center_y: int, h_img: int, w_img: int,
                     horizontal: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find missed edge when cluster vertical/horizontal edges

    :param max_len_edge_1: longest edge 1
    :param max_len_edge_2: longest edge 2
    :param center_x: center's x coordinate of polygon
    :param center_y: center's y coordinate of polygon
    :param h_img: image height
    :param w_img: image width
    :param horizontal: 1 if edges are horizontal else 0
    :return: longest edge 1 and 2 after processing
    """
    if horizontal == 0:
        vertical_dist = line_center(max_len_edge_1)[horizontal] - line_center(max_len_edge_2)[horizontal]
        if abs(vertical_dist) < w_img // 10:
            if line_center(max_len_edge_1)[0] >= center_x:
                max_len_edge_1 = max_len_edge_1 if vertical_dist >= 0 else max_len_edge_2
                max_len_edge_2 = np.array([0, 0, 0, 1], dtype=np.int32)
            else:
                max_len_edge_2 = max_len_edge_2 if vertical_dist >= 0 else max_len_edge_1
                max_len_edge_1 = np.array([w_img - 1, 0, w_img - 1, 1], dtype=np.int32)
    else:
        horizontal_dist = line_center(max_len_edge_1)[horizontal] - line_center(max_len_edge_2)[horizontal]
        if abs(horizontal_dist) < w_img // 30 * 2:
            if line_center(max_len_edge_1)[1] >= center_y:
                max_len_edge_1 = max_len_edge_1 if horizontal_dist >= 0 else max_len_edge_2
                max_len_edge_2 = np.array([0, 0, 1, 0], dtype=np.int32)
            else:
                max_len_edge_2 = max_len_edge_2 if horizontal_dist >= 0 else max_len_edge_1
                max_len_edge_1 = np.array([0, h_img - 1, 1, h_img - 1], dtype=np.int32)

    return max_len_edge_1, max_len_edge_2


def post_process(in_img: np.ndarray) -> np.ndarray:
    """
    Main post processing for card cropping

    :param in_img: input nd-array image
    :return: four vertices of input card
    """
    h_img, w_img = in_img.shape[:2]

    center_y, center_x = np.mean(np.argwhere(in_img != (0, 0, 0))[:, :2], axis=0).astype(np.int32)
    canny = cv2.Canny(in_img, 120, 255)

    '''Hough Line for edged image'''
    line_p = cv2.HoughLinesP(canny, 1, np.pi / 180, 30, None, 10, 10)

    vertical_lines, horizontal_lines = [], []
    vertical_line2center_vector, horizontal_line2center_vector = [], []

    '''Classify vertical and horizontal edges,
    then specify vectors from center of the image to them'''
    if line_p is not None:
        for i in range(0, len(line_p)):
            _line = line_p[i][0]
            if angle_with_horizontal_axis(_line) < np.pi / 6:
                horizontal_lines.append(_line)
                vector = (center_x - (_line[0] + _line[2]) // 2, center_y - (_line[1] + _line[3]) // 2)
                horizontal_line2center_vector.append(vector)
            elif angle_with_horizontal_axis(_line) > np.pi / 3:
                vertical_lines.append(_line)
                vector = (center_x - (_line[0] + _line[2]) // 2, center_y - (_line[1] + _line[3]) // 2)
                vertical_line2center_vector.append(vector)

    if len(vertical_lines) == 0:
        right_max_len_edge = np.array([w_img - 1, 0, w_img - 1, 1], dtype=np.int32)
        left_max_len_edge = np.array([0, 0, 0, 1], dtype=np.int32)

        top_max_len_edge, bot_max_len_edge = get_max_len_edges(horizontal_lines,
                                                               horizontal_line2center_vector,
                                                               horizontal=1)
        '''Check if top or bottom edge of card is missed'''
        bot_max_len_edge, top_max_len_edge = find_missed_edge(bot_max_len_edge, top_max_len_edge,
                                                              center_x, center_y, h_img, w_img,
                                                              horizontal=1)

    elif len(horizontal_lines) == 0:
        bot_max_len_edge = np.array([0, h_img - 1, 1, h_img - 1], dtype=np.int32)
        top_max_len_edge = np.array([0, 0, 1, 0], dtype=np.int32)

        left_max_len_edge, right_max_len_edge = get_max_len_edges(vertical_lines,
                                                                  vertical_line2center_vector,
                                                                  horizontal=0)

        '''Check if left or right edge of card is missed'''
        right_max_len_edge, left_max_len_edge = find_missed_edge(right_max_len_edge, left_max_len_edge,
                                                                 center_x, center_y, h_img, w_img,
                                                                 horizontal=0)

    else:
        left_max_len_edge, right_max_len_edge = get_max_len_edges(vertical_lines,
                                                                  vertical_line2center_vector,
                                                                  horizontal=0)

        top_max_len_edge, bot_max_len_edge = get_max_len_edges(horizontal_lines,
                                                               horizontal_line2center_vector,
                                                               horizontal=1)

        '''Check if left or right edge of card is missed'''
        right_max_len_edge, left_max_len_edge = find_missed_edge(right_max_len_edge, left_max_len_edge,
                                                                 center_x, center_y, h_img, w_img,
                                                                 horizontal=0)

        '''Check if top or bottom edge of card is missed'''
        bot_max_len_edge, top_max_len_edge = find_missed_edge(bot_max_len_edge, top_max_len_edge,
                                                              center_x, center_y, h_img, w_img,
                                                              horizontal=1)

    '''Calculate 4-point coordinates of card'''
    pt1 = intersection_of_2lines(right_max_len_edge, top_max_len_edge)
    pt2 = intersection_of_2lines(left_max_len_edge, top_max_len_edge)
    pt3 = intersection_of_2lines(left_max_len_edge, bot_max_len_edge)
    pt4 = intersection_of_2lines(right_max_len_edge, bot_max_len_edge)

    return np.array([pt1, pt2, pt3, pt4], dtype=np.int32)


class PostProcessing:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return post_process(img)
