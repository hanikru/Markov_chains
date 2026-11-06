import pytest
from MarkovC import Markov_chains

def test_rand_poz_range():
    """Тест: Rand_poz должен возвращать значения в диапазоне [1, size]"""
    chain = Markov_chains(size=5)
    
    # Тестируем начальное состояние
    for i in range(100):
        result = chain.Rand_poz(0)
        assert 1 <= result <= 5, f"Значение {result} не в диапазоне [1, 5]"
    
    # Тестируем переходы
    initial_state = chain.Rand_poz(0)
    for i in range(100):
        next_state = chain.Rand_poz(initial_state)
        assert 1 <= next_state <= 5, f"Значение {next_state} не в диапазоне [1, 5]"

def test_rand_poz_consistency():
    """Тест: Rand_poz должен всегда возвращать целые числа"""
    chain = Markov_chains(size=3)
    
    for i in range(50):
        result = chain.Rand_poz(0)
        assert isinstance(result, int), f"Результат {result} не является целым числом"
        assert result >= 1 and result <= 3, f"Результат {result} вне диапазона [1, 3]"

def test_markov_chains_initialization():
    """Тест: Инициализация цепей Маркова"""
    # Тест с размером
    chain = Markov_chains(size=10)
    assert chain.size == 10
    assert len(chain.vec) == 10
    assert len(chain.mat) == 10
    
    # Проверяем, что векторы и матрица заполнены
    assert all(x is not None for x in chain.vec)
    assert all(x is not None for x in chain.mat)

def test_final_distribution():
    """Тест: Финальное распределение"""
    chain = Markov_chains(size=5)
    fd = chain.Final_distribution()
    
    # Проверяем, что финальное распределение имеет правильную длину
    assert len(fd) == 5
    
    # Проверяем, что все значения неотрицательны
    for val in fd:
        assert val >= 0

def test_trajectory_generation():
    """Тест: Генерация траекторий"""
    chain = Markov_chains(size=4)
    
    # Тестируем генерацию траектории
    trajectory = chain.Mt(10)
    assert len(trajectory) == 10
    
    # Проверяем, что все значения в диапазоне [1, 4]
    for state in trajectory:
        assert 1 <= state <= 4

def test_rand_norm_vec():
    """Тест: Генерация нормализованных векторов"""
    chain = Markov_chains(size=3)
    
    # Проверяем, что векторы имеют правильную длину
    assert len(chain.vec) == 3
    assert len(chain.mat[0]) == 3
    
    # Проверяем, что сумма вероятностей равна 1
    vec_sum = sum(chain.vec)
    assert abs(vec_sum - 1.0) < 1e-10

if __name__ == "__main__":
    pytest.main([__file__, "-v"])