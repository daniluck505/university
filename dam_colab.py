import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


class DamApp():
    def make_plan(self):
        c = self.calc.contour_coordinates
        figure(figsize=(10, 2), dpi=150)
        c_x = [c[i][0] for i in range(1, 8)]
        c_y = [c[i][1] for i in range(1, 8)]
        plt.plot(c_x, c_y, color=[0.48, 0.34, 0.06])
        var9 = [c[9][0], c[9][1]]
        c_x = [c[8][0], c[9][0]]
        c_y = [c[8][1], c[9][1]]
        plt.plot(c_x, c_y, color=[0.2, 0.1, 0.9])
        c_x = [c[10][0], c[11][0]]
        c_y = [c[10][1], c[11][1]]
        plt.plot(c_x, c_y, color=[0.2, 0.1, 0.9])
        ax = plt.gca()
        ax.set_aspect('equal')
        c = self.calc.depression_curve
        tv_x = c['x'][10]
        tv_y = c['hx'][10]
        plt.plot(tv_x, tv_y, 'ro', label=f'Точка высачивания x={round(tv_x, 2)}, hx={round(tv_y, 2)}')
        plt.legend(prop={'size':7})
        c_x = c['x'][1:]
        c_x[0] = var9[0]
        c_y = c['hx'][1:]
        c_y[0] = var9[1]
        plt.plot(c_x, c_y, color=[0.2, 0.1, 0.9])
        # plt.savefig('dam.jpg', dpi=150)
        plt.show()

    def make_menu(self):
        height = self.check_input(lambda x: True if x > 0 else False, 'Высота плотины: ', 'Только положительное число')
        width = self.check_input(lambda x: True if x > 0 else False, 'Ширина гребня плотины: ', 'Только положительное число')
        z_top = self.check_input(lambda x: True if x > 0 else False, 'Заложение верхового откоса: ', 'Только положительное число')
        z_bottom = self.check_input(lambda x: True if x > 0 else False, 'Заложение низового откоса: ', 'Только положительное число')
        depth_top = self.check_input(lambda x: True if 0 < x <= height else False, 'Глубина воды в верхнем бьефе: ', 'Не больше высоты плотины')
        depth_bottom = self.check_input(lambda x: True if 0 < x <= depth_top else False, 'Глубина воды нижнем бьефе: ', 'Не больше отметки верхнего бьефа')
        thickness = self.check_input(lambda x: True if x >=0 else False, 'Толщина проницаемого слоя основания: ', 'Только положительное число')
        self.list_input_data = [height, width, z_top, z_bottom, depth_top, depth_bottom, thickness]
    
    def check_input(self, check, string, answer=None):
        try:
            value = float(input(string).replace(',', '.'))            
        except ValueError:
            print('Введите число')
            value = self.check_input(check, string, answer)
        if not check(value):
            print(answer)
            value = self.check_input(check, string, answer)
        return value

    def make_calc(self):
        self.make_menu()
        self.calc = Calculation(self.list_input_data)
        self.calc.length_filtration()
        self.calc.make_contour_coordinates()
        self.calc.make_trapezoid_area()
        self.calc.make_height_seepage()
        self.calc.make_depression_curve()
        self.make_plan()


class Calculation:
    """
    input data
    0 - высота плотины
    1 - ширина гребня плотины
    2 - заложение верхового откоса
    3 - заложение низового откоса
    4 - глубина воды в верхнем бьефе
    5 - глубина воды нижнем бьефе
    6 - толщина проницаемого слоя основания
    7 - hs

    data
    0 - расчётная длина пути фильтрации
    1 - дельта L
    2 - L_p
    3 - alpha_m
    4 -

    points {n:(x, y)}
    """
    def __init__(self, input_data):
        self.idata = input_data
        self.data = []
        
    def make_trapezoid_area(self):
        width = self.idata[0] * (self.idata[2] + self.idata[3])
        S = (2 * self.idata[1] + width) * self.idata[0] / 2
        print(f'V = {S} м3')

    def length_filtration(self):
        idata = self.make_idata_list(0, 7)
        L = (idata[0]-idata[4])*idata[2]+idata[1]+(idata[0]-idata[5])*idata[3]
        b = idata[2]/(2*idata[2]+1)
        dL = b*idata[4]
        Lp = L+dL
        alpham = idata[3]/2/(0.5+idata[3])**2

        self.data = [L, dL, Lp, alpham]

    def make_contour_coordinates(self):
        idata = self.make_idata_list(0, 7)
        points = {}
        points[3] = (0, idata[4] + idata[6])
        points[2] = (0 - idata[2]*idata[4], idata[6]) # 0? - points
        points[1] = (points[2][0]-10, idata[6])
        points[4] = (idata[2]*(idata[0]-idata[4]), idata[0]+idata[6])
        points[5] = (points[4][0]+idata[1], idata[0]+idata[6])
        points[6] = (points[5][0]+idata[3]*idata[0], idata[6])
        points[7] = (points[6][0]+10, idata[6])
        points[8] = (points[1][0], points[3][1])
        points[9] = (0, points[3][1])
        points[10] = (points[5][0]+(idata[0]-idata[5])*idata[3], idata[5]+idata[6])
        points[11] = (points[7][0], points[10][1])
        self.contour_coordinates = points

    def make_height_seepage(self):
        idata = self.make_idata_list(0, 7)
        accuracy = 11
        l_accuracy = list(range(accuracy))
        h = [(idata[4]-idata[5])*x/10 for x in l_accuracy]
        q1kf = [((idata[4]+idata[6])**2-(idata[5]+idata[6]+x)**2)/2/(self.data[2]-idata[3]*x) for x in h]
        q2akf = [x/(0.5+idata[3])*(1+idata[5]/(self.data[3]*idata[5]+x)) for x in h]
        q2bkf = [x*idata[6]/((0.5+idata[3])*x+idata[3]*idata[5]+0.4*idata[6]) for x in h]
        q2kf = [q2akf[i]+q2bkf[i] for i in l_accuracy]
        self.point_cross = self.find_cross(h, h, q1kf, q2kf)
        print(f'Удельный расход {round(self.point_cross[1], 3)}')
        # plt.plot(h, q1kf)
        # plt.plot(h, q2kf)
        # plt.xlabel('hв')
        # plt.ylabel('q/kф')
        # plt.show()

    def find_cross(self, x1, x2, y1, y2):
        idata = self.make_idata_list(0, 7)
        if idata[4] == idata[5]:
            return 0, 0

        answer = []
        x_begin = max(x1[0], x2[0])
        x_end = min(x1[-1], x2[-1])
        points1 = [t for t in zip(x1, y1) if x_begin <= t[0] <= x_end]
        points2 = [t for t in zip(x2, y2) if x_begin <= t[0] <= x_end]
        idx = 0
        while idx < len(points1) - 1:
            # y_min = min(points1[idx][1], points1[idx + 1][1])
            # y_max = max(points1[idx + 1][1], points2[idx + 1][1])
            x3 = np.linspace(points1[idx][0], points1[idx + 1][0], 1000)
            y1_new = np.linspace(points1[idx][1], points1[idx + 1][1], 1000)
            y2_new = np.linspace(points2[idx][1], points2[idx + 1][1], 1000)

            tmp_idx = np.argwhere(np.isclose(y1_new, y2_new, atol=0.01)).reshape(-1)
            if tmp_idx.all():
                if len(tmp_idx) != 0:
                    answer.append((x3[tmp_idx[0]], y2_new[tmp_idx[0]]))
            idx += 1
        print(answer[0])
        return answer[0]

    def make_depression_curve(self):
        idata = self.make_idata_list(0, 8)
        hs = self.point_cross[0]
        # print(self.point_cross[0])
        h2 = idata[5] + idata[6] + hs
        Lv = self.data[2] - idata[3]*hs
        qkf = ((idata[4]+idata[6])**2-(idata[5]+idata[6]+hs)**2)/2/(self.data[2]-idata[3]*hs)

        n = 11
        n_list = list(range(n))
        x1 = [i/(n-1)*Lv for i in n_list]
        x = [i-self.data[1] for i in x1]
        hx = [np.sqrt(2*qkf*(Lv-i)+h2**2) for i in x1]
        h1x = [idata[4]+idata[6] if x[i]<0 else hx[i] for i in n_list]
        hx_T = [i - idata[6] for i in hx]
        print(f'Высота высачивания = {round(hx[-1] - idata[6] - idata[5], 2)}')
        self.depression_curve = {'hx':hx, 'x':x}
        # self.depression_curve = {'hx':h1x, 'x':x}
        print('Координаты кривой депрессии: ')
        print(f'x: {[round(i, 2) for i in x]}')
        print(f'hx: {[round(i, 2) for i in hx]}')

    def make_idata_list(self, a, b):
        idata = [float(x) for x in self.idata[a:b]]
        return idata


if __name__ == '__main__':
    Dam = DamApp()
    Dam.make_calc()
