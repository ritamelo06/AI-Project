# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 131:
# 106507 Martim Afonso
# 107294 Rita Melo

import numpy as np
import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""
    size_n = 0

    def __init__(self, data):
        self.data = data

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.data[row][col]

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        row_above = row - 1
        row_below = row + 1
        if row_above < 0:
            return None, self.get_value(row_below, col)
        elif row_below > self.size_n - 1:
            return self.get_value(row_above, col), None
        else:
            return self.get_value(row_above, col), self.get_value(row_below, col)


    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        col_left = col - 1
        col_right = col + 1
        if col_left < 0:
            return None, self.get_value(row, col_left)
        elif col_right > self.size_n - 1:
            return self.get_value(row, col_left), None
        else:
            return self.get_value(row, col_left), self.get_value(row, col_right)

    def print_board(self) -> str:
        for row in range(self.size_n):
            print(" ".join(self.data[row]))

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """

        lines = sys.stdin.readlines()
        Board.size_n = len(lines)
        processed_data = [line.strip().split('\t') for line in lines if line.strip()]
        np_array = np.array(processed_data)   #converter a matriz num nparray
        return Board(np_array)

    # TODO: outros metodos da classe


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = PipeManiaState(board) 

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        last_idx = state.board.size_n - 1   

        for row in range(state.board.size_n):
            for col in range(state.board.size_n):
                # Cantos
                # canto superior esquerdo
                if row == 0 and col == 0:
                    top_left = state.board.get_value(0,0)
                    if top_left == 'VB':
                        continue
                    if top_left in ['VE', 'FB', 'FE', 'VC']:
                        actions.append((0, 0, False))
                    if top_left in ['VD', 'FD', 'FC', 'VC']:
                        actions.append((0, 0, True))
                # canto inferior esquerdo
                elif row == last_idx and col == 0:
                    bottom_left = state.board.get_value(last_idx, 0)
                    if bottom_left == 'VD':
                        continue
                    if bottom_left in ['VB', 'FB', 'FD', 'VE']:
                        actions.append((last_idx, 0, False))
                    if bottom_left in ['VC', 'FC', 'FE', 'VE']:
                        actions.append((last_idx, 0, True))

                # canto inferior direito
                elif row == last_idx and col == last_idx:
                    bottom_right = state.board.get_value(last_idx, last_idx)
                    if bottom_right == 'VC':
                        continue
                    if bottom_right in ['VD', 'FC', 'FD', 'VB']:
                        actions.append((last_idx, last_idx, False))
                    if bottom_right in ['VE', 'FB', 'FE', 'VB']:
                        actions.append((last_idx, last_idx, True))

                # canto superior direito
                elif row == 0 and col == last_idx:
                    top_right = state.board.get_value(0, last_idx)
                    if top_right == 'VE':
                        continue
                    if top_right in ['VC', 'FC', 'FE', 'VD']:
                        actions.append((0, last_idx, False))
                    if top_right in ['VB', 'FB', 'FD', 'VD']:
                        actions.append((0, last_idx, True))

                # Bordas
                elif row == 0 or col == 0 or row == last_idx or col == last_idx:
                    pipe = state.board.get_value(row, col)
                    # borda cima
                    if row == 0:
                        if pipe in ['FC', 'FB', 'FE', 'BC', 'BE', 'VC', 'VE', 'LV']:
                            actions.append((row, col, False))
                        if pipe in ['FC', 'FB', 'FD', 'BC', 'BD', 'VB', 'VD', 'LV']:
                            actions.append((row, col, True))
                    # borda esquerda
                    elif col == 0:
                        if pipe in ['FB', 'FE', 'FD', 'BB', 'BE', 'VB', 'VE', 'LH']:
                            actions.append((row, col, False))
                        if pipe in ['FC', 'FE', 'FD', 'BC', 'BE', 'VC', 'VD', 'LH']:
                            actions.append((row, col, True))
                    # borda baixo
                    elif row == last_idx:
                        if pipe in ['FC', 'FB', 'FD', 'BB', 'BD', 'VB', 'VD', 'LV']:
                            actions.append((row, col, False))
                        if pipe in ['FC', 'FB', 'FE', 'BB', 'BE', 'VC', 'VE', 'LV']:
                            actions.append((row, col, True))
                    # borda direita
                    elif col == last_idx:
                        if pipe in ['FC', 'FE', 'FD', 'BC', 'BD', 'VC', 'VD', 'LH']:
                            actions.append((row, col, False))
                        if pipe in ['FB', 'FE', 'FD', 'BB', 'BD', 'VB', 'VE', 'LH']:
                            actions.append((row, col, True))

                # Interior
                else:
                    actions.append((row, col, True))
                    actions.append((row, col, False))
        return actions

    # FUNÇÃO PARA COPIAR O BOARD (aula pratica)

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    board = Board.parse_instance()
    board.print_board()
    problem = PipeMania(board)
    initial_state = PipeManiaState(board)
    print(problem.actions(initial_state))

