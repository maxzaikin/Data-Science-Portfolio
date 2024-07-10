Отлично, давайте рассмотрим более детально шаги обратного распространения (backpropagation) и обновления весов. Мы пройдем через каждый шаг, начиная с вычисления ошибок (градиентов), накопления их через временные шаги и обновления весов.

### Повторим шаги прямого распространения (forward propagation)

Для простоты, возьмем только два шага последовательности "hel" для входа и "ell" для цели.

#### Шаг 1: Вход 'h'

1. **Входной вектор:** \( x_1 = [1, 0, 0, 0, 0] \)
2. **Скрытое состояние:**

\[
h_1 = \tanh(W_{xh} \cdot x_1 + W_{hh} \cdot h_0)
\]

\[
h_1 = \tanh(\begin{bmatrix}
0.1 & 0.2 & 0.3 \\
0.4 & 0.5 & 0.6 \\
0.7 & 0.8 & 0.9 \\
0.1 & 0.1 & 0.1 \\
0.1 & 0.1 & 0.1
\end{bmatrix} \cdot \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix})
\]

\[
h_1 = \tanh(\begin{bmatrix} 0.1 \\ 0.4 \\ 0.7 \\ 0.1 \\ 0.1 \end{bmatrix})
\]

\[
h_1 = \begin{bmatrix} 0.0997 \\ 0.3799 \\ 0.6044 \\ 0.0997 \\ 0.0997 \end{bmatrix}
\]

3. **Выходной вектор:**

\[
y_1 = W_{hy} \cdot h_1
\]

\[
y_1 = \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.6 & 0.7 & 0.8 & 0.9 & 1.0 \\
1.1 & 1.2 & 1.3 & 1.4 & 1.5
\end{bmatrix} \cdot \begin{bmatrix} 0.0997 \\ 0.3799 \\ 0.6044 \end{bmatrix}
\]

\[
y_1 = \begin{bmatrix}
0.1 \cdot 0.0997 + 0.2 \cdot 0.3799 + 0.3 \cdot 0.6044 \\
0.4 \cdot 0.0997 + 0.5 \cdot 0.3799 + 0.6 \cdot 0.6044 \\
0.7 \cdot 0.0997 + 0.8 \cdot 0.3799 + 0.9 \cdot 0.6044 \\
0.1 \cdot 0.0997 + 0.1 \cdot 0.3799 + 0.1 \cdot 0.6044 \\
0.1 \cdot 0.0997 + 0.1 \cdot 0.3799 + 0.1 \cdot 0.6044
\end{bmatrix}
\]

\[
y_1 = \begin{bmatrix}
0.4998 \\
0.9998 \\
1.4997 \\
0.9998 \\
1.4997
\end{bmatrix}
\]

#### Шаг 2: Вход 'e'

1. **Входной вектор:** \( x_2 = [0, 1, 0, 0, 0] \)
2. **Скрытое состояние:**

\[
h_2 = \tanh(W_{xh} \cdot x_2 + W_{hh} \cdot h_1)
\]

\[
h_2 = \tanh(\begin{bmatrix}
0.1 & 0.2 & 0.3 \\
0.4 & 0.5 & 0.6 \\
0.7 & 0.8 & 0.9 \\
0.1 & 0.1 & 0.1 \\
0.1 & 0.1 & 0.1
\end{bmatrix} \cdot \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix}
0.1 & 0.2 & 0.3 \\
0.4 & 0.5 & 0.6 \\
0.7 & 0.8 & 0.9
\end{bmatrix} \cdot \begin{bmatrix} 0.0997 \\ 0.3799 \\ 0.6044 \end{bmatrix})
\]

\[
h_2 = \tanh(\begin{bmatrix}
0.2 \\
0.5 \\
0.8 \\
0.1 \\
0.1
\end{bmatrix} + \begin{bmatrix}
0.4099 \\
1.5296 \\
2.6493
\end{bmatrix})
\]

\[
h_2 = \tanh(\begin{bmatrix}
0.6099 \\
2.0296 \\
3.4493
\end{bmatrix})
\]

3. **Выходной вектор:**

\[
y_2 = W_{hy} \cdot h_2
\]

\[
y_2 = \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.6 & 0.7 & 0.8 & 0.9 & 1.0 \\
1.1 & 1.2 & 1.3 & 1.4 & 1.5
\end{bmatrix} \cdot \begin{bmatrix} 0.0997 \\ 0.3799 \\ 0.6044 \end{bmatrix}
\]

\[
y_2 = \begin{bmatrix}
0.4998 \\
0.9998 \\
1.4997 \\
0.9998 \\
1.4997
\end{bmatrix}
\]

### Функция потерь

Целевая последовательность для двух шагов:

- 'e' -> [0, 1, 0, 0, 0]
- 'l' -> [0, 0, 1, 0, 0]

Функция потерь для первого шага:

\[
\text{Loss}_1 = -\sum_{i} t_i \log(y_{1i})
\]

\[
\text{Loss}_1 = - (0 \cdot \log(0.4998) + 1 \cdot \log(0.9998) + 0 \cdot \log(1.4997) + 0 \cdot \log(0.9998) + 0 \cdot \log(1.4997))
\]

\[
\text{Loss}_1 = - \log(0.9998)
\]

Для второго шага:

\[
\text{Loss}_2 = -\sum_{i} t_i \log(y_{2i})
\]

\[
\text{Loss}_2 = - (0 \cdot \log(0.4998) + 0 \cdot \log(0.9998) + 1 \cdot \log(1.4997) + 0 \cdot \log(0.9998) + 0 \cdot \log(1.4997))
\]

\[
\text{Loss}_2 = - \log(1.4997)
\]

### Обратное распространение (Backpropagation)

#### Градиенты для второго шага:

\[
\frac{\partial \text{Loss}_2}{\partial W_{hy}} = (y_2 - t_2) \cdot h_2^T
\]

\[
\frac{\partial \text{Loss}_2}{\partial W_{hh}} = \frac{\partial \text{Loss}_2}{\partial h_2} \cdot \frac{\partial h_2}{\partial W_{hh}}
\]

\[
\frac{\partial \text{Loss}_2}{\partial W_{xh}} = \frac{\partial \text{Loss}_2}{\partial h_2} \cdot \frac{\partial h_2}{\partial W_{xh}}
\]

#### Градиенты для первого шага:

\[
\frac{\partial \text{Loss}_1}{\partial W_{hy}} = (y_1 - t_1) \cd

ot h_1^T
\]

\[
\frac{\partial \text{Loss}_1}{\partial W_{hh}} = \frac{\partial \text{Loss}_1}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_{hh}}
\]

\[
\frac{\partial \text{Loss}_1}{\partial W_{xh}} = \frac{\partial \text{Loss}_1}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_{xh}}
\]

### Обновление весов

Весы обновляются с использованием градиентного спуска:

\[
W = W - \alpha \frac{\partial \text{Loss}}{\partial W}
\]

где \( \alpha \) - скорость обучения (learning rate).

После этих шагов модель корректирует свои веса и улучшает свои предсказания. Таким образом, на каждой итерации модель постепенно учится предсказывать правильные символы на основе входных данных.
