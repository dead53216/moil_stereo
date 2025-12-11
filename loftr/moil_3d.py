import math
import time
from typing import Tuple

from sympy import Eq, Symbol, solve


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

    @staticmethod
    def get_unknown_3d_coord_and_unknown_symbol(
        vector: Tuple[float, float, float],
        cam_3d_coord: Tuple[float, float, float],
        symbol_name: str,
    ) -> Tuple[Tuple[float, float, float], Symbol]:
        vector_x = vector[0]
        vector_y = vector[1]
        vector_z = vector[2]

        cam_3d_coord_x = cam_3d_coord[0]
        cam_3d_coord_y = cam_3d_coord[1]
        cam_3d_coord_z = cam_3d_coord[2]

        unknown_symbol = Symbol(symbol_name)
        unknown_coord = (
            vector_x * unknown_symbol + cam_3d_coord_x,
            vector_y * unknown_symbol + cam_3d_coord_y,
            vector_z * unknown_symbol + cam_3d_coord_z,
        )

        return unknown_coord, unknown_symbol

    @staticmethod
    def solve_unknown_symbol_m_n(
        left_vector: Tuple[float, float, float],
        right_vector: Tuple[float, float, float],
        unknown_coord_p: Tuple[float, float, float],
        unknown_coord_q: Tuple[float, float, float],
        unknown_symbol_m: Symbol,
        unknown_symbol_n: Symbol,
    ) -> Tuple[Symbol, Symbol]:
        vector_pq_x = unknown_coord_q[0] - unknown_coord_p[0]
        vector_pq_y = unknown_coord_q[1] - unknown_coord_p[1]
        vector_pq_z = unknown_coord_q[2] - unknown_coord_p[2]

        left_vector_x = left_vector[0]
        left_vector_y = left_vector[1]
        left_vector_z = left_vector[2]

        right_vector_x = right_vector[0]
        right_vector_y = right_vector[1]
        right_vector_z = right_vector[2]

        eq1 = Eq(
            (
                vector_pq_x * left_vector_x
                + vector_pq_y * left_vector_y
                + vector_pq_z * left_vector_z
            ),
            0,
        )

        eq2 = Eq(
            (
                vector_pq_x * right_vector_x
                + vector_pq_y * right_vector_y
                + vector_pq_z * right_vector_z
            ),
            0,
        )

        ans_s_t = solve((eq1, eq2), (unknown_symbol_m, unknown_symbol_n))

        unknown_symbol_m = ans_s_t[unknown_symbol_m]
        unknown_symbol_n = ans_s_t[unknown_symbol_n]

        return unknown_symbol_m, unknown_symbol_n

    @staticmethod
    def calculate_closest_2_3d_coord_of_2_view_line(
        left_vector: Tuple[float, float, float],
        right_vector: Tuple[float, float, float],
        cam_3d_coord_l: Tuple[float, float, float],
        cam_3d_coord_r: Tuple[float, float, float],
        n: Symbol,
        m: Symbol,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        point_p = (
            left_vector[0] * n + cam_3d_coord_l[0],
            left_vector[1] * n + cam_3d_coord_l[1],
            left_vector[2] * n + cam_3d_coord_l[2],
        )

        point_q = (
            right_vector[0] * m + cam_3d_coord_r[0],
            right_vector[1] * m + cam_3d_coord_r[1],
            right_vector[2] * m + cam_3d_coord_r[2],
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
        # 用於儲存每個步驟的時間
        timing_results = {}
        total_start = time.perf_counter()

        # Step 1: 計算向量 (vector_l, vector_r)
        start = time.perf_counter()
        vector_l = Moil3dAlgorithm.al_ba_2_vector(alpha_l, beta_l)
        vector_r = Moil3dAlgorithm.al_ba_2_vector(alpha_r, beta_r)
        end = time.perf_counter()
        timing_results["1. al_ba_2_vector (l & r)"] = end - start

        # Step 2: 取得未知 3D 座標和未知符號 (point_p, m, point_q, n)
        start = time.perf_counter()
        point_p, m = Moil3dAlgorithm.get_unknown_3d_coord_and_unknown_symbol(
            vector_l, cam_3d_coord_l, "m"
        )

        point_q, n = Moil3dAlgorithm.get_unknown_3d_coord_and_unknown_symbol(
            vector_r, cam_3d_coord_r, "n"
        )
        end = time.perf_counter()
        timing_results["2. get_unknown_3d_coord_and_unknown_symbol"] = end - start

        # Step 3: 解未知符號 m, n (使用 Sympy 的 solve)
        start = time.perf_counter()
        m_solved, n_solved = Moil3dAlgorithm.solve_unknown_symbol_m_n(
            vector_l, vector_r, point_p, point_q, m, n
        )
        end = time.perf_counter()
        timing_results["3. solve_unknown_symbol_m_n (Sympy Solve)"] = end - start

        # 將 Sympy 的 Symbol 轉換為浮點數
        m_float = float(m_solved)
        n_float = float(n_solved)

        # Step 4: 計算最接近的兩個 3D 座標 (point_p, point_q)
        start = time.perf_counter()
        # 注意: 這裡傳入的 n 和 m 已經是解出來的值 (n_float, m_float)
        point_p_final = (
            vector_l[0] * n_float + cam_3d_coord_l[0],
            vector_l[1] * n_float + cam_3d_coord_l[1],
            vector_l[2] * n_float + cam_3d_coord_l[2],
        )

        point_q_final = (
            vector_r[0] * m_float + cam_3d_coord_r[0],
            vector_r[1] * m_float + cam_3d_coord_r[1],
            vector_r[2] * m_float + cam_3d_coord_r[2],
        )
        end = time.perf_counter()
        timing_results["4. calculate_closest_2_3d_coord_of_2_view_line"] = end - start

        # Step 5: 計算中點 (mid_point)
        start = time.perf_counter()
        mid_point = Moil3dAlgorithm.get_mid_point(point_p_final, point_q_final)
        end = time.perf_counter()
        timing_results["5. get_mid_point"] = end - start

        total_end = time.perf_counter()
        total_time = total_end - total_start

        # 輸出計時結果
        print("\n=== nearest_mid_3d_coord 耗時分析 ===")
        print(f"總執行時間: {total_time:.6f} 秒")
        for step, duration in timing_results.items():
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"- {step:<40}: {duration:.6f} 秒 ({percentage:.2f}%)")
        print("====================================\n")

        return mid_point

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
        left_alpha, left_beta = left_alpha, left_beta

        right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p1, 1)
        right_alpha, right_beta = right_alpha, right_beta

        point_p = Moil3dAlgorithm.nearest_mid_3d_coord(
            left_alpha,
            left_beta,
            right_alpha,
            right_beta,
            l_cam_3d_coord,
            r_cam_3d_coord,
        )

        left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_p2, 1)
        left_alpha, left_beta = left_alpha, left_beta

        right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p2, 1)
        right_alpha, right_beta = right_alpha, right_beta

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
