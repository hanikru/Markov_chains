import numpy as np
import random

class Markov_chains:
    def __init__(self, mat: list = None, vec: list = None, size: int = 0):
        if mat is not None and vec is not None and size == 0:
            self.mat = mat
            self.vec = vec
            self.size = len(mat)
            self.tra = []
            
        elif mat is None and vec is None and size > 0:
            self.size = size
            self.tra = []
            self.mat = [None] * size 
            self.vec = [None] * size
            self.Mtp()

        else:
            print("Класс инициализирован не корректно")


    def Find_state(self, time):
        if time < 0:
            raise ValueError("time must be non-negative")

        temp_mat = self.mat
        result = self.vec.copy()
        for _ in range(time):
            result = result @ temp_mat
        return result

    def Final_distribution(self):
        mat = np.array(self.mat)

        A = mat.T - np.eye(self.size)
        A[-1, :] = 1

        b = np.zeros(self.size)
        b[-1] = 1

        result = np.linalg.solve(A, b)
        return result

    def Rand_norm_vec(self):
        rand_vec = [random.random() for _ in range(self.size - 1)]
        rand_vec.insert(0, 0)
        rand_vec.insert(-1, 1)
        rand_vec = sorted(rand_vec)

        result = [round(rand_vec[i+1] - rand_vec[i], 10) for i in range(self.size)]

        if sum(result) > 1.0:
            result[-1] = result[-1] - (sum(result) - 1.0)

        return result

    # моделирование переходной вероятности
    def Mtp(self):
        self.vec = self.Rand_norm_vec()
        for i in range(self.size):
            self.mat[i] = self.Rand_norm_vec()

    def Rand_poz(self, idx):
        pred = random.random()

        if idx == 0:
            vec_temp = [sum(self.vec[:i]) for i in range(len(self.vec) + 1)]
        elif idx > 0:
            vec_temp = [sum(self.mat[idx - 1][:i]) for i in range(len(self.mat) + 1)]
        else: 
            print("номер состояния не может быть отрицательным")
            return None
        
        for i in range(self.size):
            if vec_temp[i] < pred <= vec_temp[i + 1]:
                self.tra.append(i+1)
                return i+1

        return 1

    # моделирование траектории цепи маркова
    def Mt(self, step):
        self.tra = []
        
        start = self.Rand_poz(0)
        result = [start]

        for _ in range(1, step):
            next_state = self.Rand_poz(start)
            result.append(next_state)
            start = next_state

        return result


    # вероятность траектории
    def Pt(self, tra):
        idx = np.array(tra) - 1
        mat = np.asarray(self.mat) 
        return self.vec[idx[0]] * np.prod(mat[idx[:-1], idx[1:]])