import math
from typing import Tuple

import numpy as np  # 引入 NumPy

# 移除 from sympy import Eq, Symbol, solve


class Moil3dAlgorithm:
    @staticmethod
    def _beta_moil2cartesian(beta_moil: float) -> float:
        beta_cartesian = beta_moil % 360
        beta_cartesian = 90 - beta_cartesian

        if beta_cartesian < 0:
            return beta_cartesian + 360
        else:
            return beta_cartesian

    @staticmethod
    def al_ba_2_vector_x(alpha: float, beta: float) -> float:
        beta = Moil3dAlgorithm._beta_moil2cartesian(beta)
        return math.sin(math.radians(alpha)) * math.cos(math.radians(beta))

    @staticmethod
    def al_ba_2_vector_y(alpha: float, beta: float) -> float:
        beta = Moil3dAlgorithm._beta_moil2cartesian(beta)
        return math.sin(math.radians(alpha)) * math.sin(math.radians(beta))

    @staticmethod
    def al_2_vector_z(alpha: float) -> float:
        return math.cos(math.radians(alpha))

    @staticmethod
    def al_ba_2_vector(alpha: float, beta: float) -> Tuple[float, float, float]:
        vector_x = Moil3dAlgorithm.al_ba_2_vector_x(alpha, beta)
        vector_y = Moil3dAlgorithm.al_ba_2_vector_y(alpha, beta)
        vector_z = Moil3dAlgorithm.al_2_vector_z(alpha)

        return vector_x, vector_y, vector_z

    # 移除 al_ba_2_vector (get_unknown_3d_coord_and_unknown_symbol)

    @staticmethod
    def solve_unknown_symbol_m_n_numerical(
        vector_l: Tuple[float, float, float],
        vector_r: Tuple[float, float, float],
        cam_3d_coord_l: Tuple[float, float, float],
        cam_3d_coord_r: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """
        使用 NumPy 數值方法求解 m 和 n (兩射線最短距離參數)。

        Args:
            vector_l: 左相機的視線向量 V_L.
            vector_r: 右相機的視線向量 V_R.
            cam_3d_coord_l: 左相機 3D 座標 C_L.
            cam_3d_coord_r: 右相機 3D 座標 C_R.

        Returns:
            Tuple[float, float]: 參數 (m, n).
        """

        # 轉換為 NumPy 陣列
        vl = np.array(vector_l)
        vr = np.array(vector_r)
        cl = np.array(cam_3d_coord_l)
        cr = np.array(cam_3d_coord_r)

        # C_LR = C_L - C_R (這裡使用 C_L - C_R，以匹配標準最短距離公式的矩陣設定)
        cl_cr = cl - cr

        # 計算點積
        vl_dot_vl = np.dot(vl, vl)
        vr_dot_vr = np.dot(vr, vr)
        vl_dot_vr = np.dot(vl, vr)  # 也是 vr_dot_vl

        # 係數矩陣 A (針對變量 m, n)
        # 公式形式: m(VL·VL) - n(VR·VL) = (CR - CL)·VL
        #          m(VL·VR) - n(VR·VR) = (CR - CL)·VR
        # 為了匹配標準 NumPy 求解形式: Ax = B (其中 x = [m, n])
        # 我們使用 (CR - CL)·V 轉換為 (CL - CR)·V 的負值，或者直接將 C_LR 定義為 CL - CR

        A = np.array([[vl_dot_vl, -vl_dot_vr], [vl_dot_vr, -vr_dot_vr]])

        # 常數向量 B
        # B = [ (CL - CR)·VL, (CL - CR)·VR ]
        B = np.array([np.dot(cl_cr, vl), np.dot(cl_cr, vr)])

        try:
            # 求解 m 和 n
            m_n_solved = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # 兩射線平行或重合，矩陣為奇異矩陣
            # 在實際應用中可能需要更複雜的處理，這裡簡單返回 0.0, 0.0 或拋出錯誤
            raise ValueError("LinAlgError: Rays are parallel or too close to parallel.")

        # 返回 m 和 n
        return float(m_n_solved[0]), float(m_n_solved[1])

    # 移除 solve_unknown_symbol_m_n

    @staticmethod
    def calculate_closest_2_3d_coord_of_2_view_line(
        left_vector: Tuple[float, float, float],
        right_vector: Tuple[float, float, float],
        cam_3d_coord_l: Tuple[float, float, float],
        cam_3d_coord_r: Tuple[float, float, float],
        m: float,  # 這裡 m 和 n 現在是 float 而非 Symbol
        n: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        # 注意: 根據最短距離公式的慣例，參數通常是 n 用於 V_L (左射線)，m 用於 V_R (右射線)。
        # 為了與您的原始程式碼一致 (原始代碼 m 對應 point_p/left_vector, n 對應 point_q/right_vector)，
        # 我們讓 m 驅動 point_p (左) 的距離，n 驅動 point_q (右) 的距離，這與 solve_unknown_symbol_m_n_numerical 的返回順序有關。

        # 根據 solve_unknown_symbol_m_n_numerical 的返回順序，m 是左射線的參數，n 是右射線的參數

        # point_p (左射線上的點)
        point_p = (
            left_vector[0] * m + cam_3d_coord_l[0],
            left_vector[1] * m + cam_3d_coord_l[1],
            left_vector[2] * m + cam_3d_coord_l[2],
        )

        # point_q (右射線上的點)
        point_q = (
            right_vector[0] * n + cam_3d_coord_r[0],
            right_vector[1] * n + cam_3d_coord_r[1],
            right_vector[2] * n + cam_3d_coord_r[2],
        )

        return point_p, point_q

    @staticmethod
    def get_mid_point(
        point_p: Tuple[float, float, float], point_q: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        mid_point = (
            (point_p[0] + point_q[0]) / 2,
            (point_p[1] + point_q[1]) / 2,
            (point_p[2] + point_q[2]) / 2,
        )

        return mid_point

    @staticmethod
    def nearest_mid_3d_coord(
        alpha_l: float,
        beta_l: float,
        alpha_r: float,
        beta_r: float,
        cam_3d_coord_l: Tuple[float, float, float],
        cam_3d_coord_r: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        # 1. 計算左右相機的視線向量
        vector_l = Moil3dAlgorithm.al_ba_2_vector(alpha_l, beta_l)
        vector_r = Moil3dAlgorithm.al_ba_2_vector(alpha_r, beta_r)

        # 2. 使用 NumPy 數值求解 m 和 n
        # m: 左射線的參數 (point_p)
        # n: 右射線的參數 (point_q)
        m, n = Moil3dAlgorithm.solve_unknown_symbol_m_n_numerical(
            vector_l, vector_r, cam_3d_coord_l, cam_3d_coord_r
        )

        # 3. 計算兩條射線上最接近的兩個 3D 座標點 (point_p 和 point_q)
        point_p, point_q = Moil3dAlgorithm.calculate_closest_2_3d_coord_of_2_view_line(
            vector_l, vector_r, cam_3d_coord_l, cam_3d_coord_r, m, n
        )

        # 4. 計算中點
        mid_point = Moil3dAlgorithm.get_mid_point(point_p, point_q)

        return mid_point

    # ... (quick_3d_measure 和 single_3d_coordinate 方法保持不變, 它們依賴 nearest_mid_3d_coord)

    @staticmethod
    def quick_3d_measure(
        l_moildev,
        r_moildev,
        l_cam_3d_coord: Tuple[float, float, float],
        r_cam_3d_coord: Tuple[float, float, float],
        l_cam_p1: Tuple[int, int],
        l_cam_p2: Tuple[int, int],
        r_cam_p1: Tuple[int, int],
        r_cam_p2: Tuple[int, int],
    ) -> float:
        """
        :param l_moildev: Moildev object (cam_l)
        :param r_moildev: Moildev object (cam_r)
        :param l_cam_3d_coord: Tuple[x: float, y: float, z: float] 3d coord (cam_l)
        :param r_cam_3d_coord: Tuple[x: float, y: float, z: float] 3d coord (cam_r)
        :param l_cam_p1: Tuple[x: int, y: int] pixel coord "p1" (cam_l)
        :param l_cam_p2: Tuple[x: int, y: int] pixel coord "p2" (cam_r)
        :param r_cam_p1: Tuple[x: int, y: int] pixel coord "p1" (cam_l)
        :param r_cam_p2: Tuple[x: int, y: int] pixel coord "p2" (cam_r)
        :return: float
        """

        left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_p1, 1)

        right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p1, 1)

        point_p = Moil3dAlgorithm.nearest_mid_3d_coord(
            left_alpha,
            left_beta,
            right_alpha,
            right_beta,
            l_cam_3d_coord,
            r_cam_3d_coord,
        )

        left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_p2, 1)

        right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p2, 1)

        point_q = Moil3dAlgorithm.nearest_mid_3d_coord(
            left_alpha,
            left_beta,
            right_alpha,
            right_beta,
            l_cam_3d_coord,
            r_cam_3d_coord,
        )
        distance = math.sqrt(
            pow(point_p[0] - point_q[0], 2)
            + pow(point_p[1] - point_q[1], 2)
            + pow(point_p[2] - point_q[2], 2)
        )
        return round(distance, 2)

    @staticmethod
    def single_3d_coordinate(
        l_moildev,
        r_moildev,
        l_cam_3d_coord: Tuple[float, float, float],
        r_cam_3d_coord: Tuple[float, float, float],
        l_cam_pixel_coord: Tuple[int, int],
        r_cam_pixel_coord: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        """
        :param l_moildev: Moildev object (cam_l)
        :param r_moildev: Moildev object (cam_r)
        :param l_cam_3d_coord: Tuple[x: float, y: float, z: float] 3d coord (cam_l)
        :param r_cam_3d_coord: Tuple[x: float, y: float, z: float] 3d coord (cam_r)
        :param l_cam_pixel_coord: Tuple[x: int, y: int] pixel coord (cam_l)
        :param r_cam_pixel_coord: Tuple[x: int, y: int] pixel coord (cam_r)
        :return: Tuple

        This method calculates the single coordinate of the target on the image.
        Only the single corresponding pixel coordinates on the left and right images need to be selected.
        """

        left_alpha, left_beta = l_moildev.get_alpha_beta(*l_cam_pixel_coord, 1)
        left_alpha, left_beta = round(left_alpha, 1), round(left_beta, 1)

        right_alpha, right_beta = r_moildev.get_alpha_beta(*r_cam_pixel_coord, 1)
        right_alpha, right_beta = round(right_alpha, 1), round(right_beta, 1)

        return Moil3dAlgorithm.nearest_mid_3d_coord(
            left_alpha,
            left_beta,
            right_alpha,
            right_beta,
            l_cam_3d_coord,
            r_cam_3d_coord,
        )
