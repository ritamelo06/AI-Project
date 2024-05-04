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
                # canto superior esquerdo
                if row == 0 and col == 0:     
                    top_left = state.board.get_value(0,0)
                    if top_left == 'VB':
                        continue
                    elif top_left == 'VE':
                        actions.append((0, 0, False))
                    elif top_left == 'VD':
                        actions.append((0, 0, True))
                    elif top_left == 'FD':
                        actions.append((0, 0, True))
                    elif top_left == 'FB':
                        actions.append((0, 0, False))
                    elif top_left == 'FC':
                        actions.append((0, 0, True))
                    elif top_left == 'FB':
                        actions.append((0, 0, False))
                    elif top_left == 'VC':
                        actions.append((0, 0, False))
                        actions.append((0, 0, True))
                
                # canto inferior esquerdo    
                elif row == last_idx and col == 0:
                    bottom_left = state.board.get_value(last_idx, 0)
                    if bottom_left == 'VD':
                        continue
                    elif bottom_left == 'VB':
                        actions.append((last_idx, 0, False))
                    elif bottom_left == 'VC':
                        actions.append((last_idx, 0, True))
                    elif bottom_left == 'FC':
                        actions.append((last_idx, 0, True))
                    elif bottom_left == 'FB':
                        actions.append((last_idx, 0, False))
                    elif bottom_left == 'FE':
                        actions.append((last_idx, 0, True))
                    elif bottom_left == 'FD':
                        actions.append((last_idx, 0, False))
                    elif bottom_left == 'VE':
                        actions.append((last_idx, 0, False))
                        actions.append((last_idx, 0, True))
                
                # canto inferior direito    
                elif row == last_idx and col == last_idx:
                    bottom_right = state.board.get_value(last_idx, last_idx)
                    if bottom_right == 'VC':
                        continue
                    elif bottom_right == 'VD':
                        actions.append((last_idx, last_idx, False))
                    elif bottom_right == 'VE':
                        actions.append((last_idx, last_idx, True))
                    elif bottom_right == 'FC':
                        actions.append((last_idx, last_idx, False))
                    elif bottom_right == 'FB':
                        actions.append((last_idx, last_idx, True))
                    elif bottom_right == 'FE':
                        actions.append((last_idx, last_idx, True))
                    elif bottom_right == 'FD':
                        actions.append((last_idx, last_idx, False))
                    elif bottom_right == 'VB':
                        actions.append((last_idx, last_idx, False))
                        actions.append((last_idx, last_idx, True))
                
                # canto superior direito
                elif row == 0 and col == last_idx:
                    top_right = state.board.get_value(0, last_idx)
                    if top_right == 'VE':
                        continue
                    elif top_right == 'VC':
                        actions.append((0, last_idx, False))
                    elif top_right == 'VB':
                        actions.append((0, last_idx, True))
                    elif top_right == 'FC':
                        actions.append((0, last_idx, False))
                    elif top_right == 'FB':
                        actions.append((0, last_idx, True))
                    elif top_right == 'FE':
                        actions.append((0, last_idx, False))
                    elif top_right == 'FD':
                        actions.append((0, last_idx, True))
                    elif top_right == 'VD':
                        actions.append((0, last_idx, False)) 
                        actions.append((0, last_idx, True)) 

                # fronteira da esquerda
                elif col == 0:
                    pipe = state.board.get_value(row, 0)
                    if pipe == 'BD' or pipe == 'LV':
                        continue
                    elif pipe == 'BB':
                        actions.append((row, 0, False))
                    elif pipe == 'BC':
                        actions.append((row, 0, True))
                    elif pipe == 'FB':
                        actions.append((row, 0, False))
                    elif pipe == 'FC':
                        actions.append((row, 0, True))
                    elif pipe == 'VB':
                        actions.append((row, 0, False))
                    elif pipe == 'VC':
                        actions.append((row, 0, True))
                    elif pipe == 'VE':
                        actions.append((row, 0, False))
                    elif pipe == 'VD':
                        actions.append((row, 0, True))
                    else:
                        actions.append((row, 0, False))
                        actions.append((row, 0, True))

                # fronteira da direita
                elif col == last_idx:
                    pipe = state.board.get_value(row, last_idx)
                    if pipe == 'BE' or pipe == 'LV':
                        continue
                    elif pipe == 'BB':
                        actions.append((row, last_idx, True))
                    elif pipe == 'BC':
                        actions.append((row, last_idx, False))
                    elif pipe == 'FB':
                        actions.append((row, last_idx, True))
                    elif pipe == 'FC':
                        actions.append((row, last_idx, False))
                    elif pipe == 'VB':
                        actions.append((row, last_idx, True))
                    elif pipe == 'VC':
                        actions.append((row, last_idx, False))
                    elif pipe == 'VE':
                        actions.append((row, last_idx, True))
                    elif pipe == 'VD':
                        actions.append((row, last_idx, False))
                    else:
                        actions.append((row, last_idx, True))
                        actions.append((row, last_idx, False))

                # fronteira de cima
                elif row == 0:
                    pipe = state.board.get_value(0, col)
                    if pipe == 'BB' or pipe == 'LH':
                        continue
                    elif pipe == 'BE':
                        actions.append((0, col, False))
                    elif pipe == 'BD':
                        actions.append((0, col, True))
                    elif pipe == 'FE':
                        actions.append((0, col, False))
                    elif pipe == 'FD':
                        actions.append((0, col, True))
                    elif pipe == 'VB':
                        actions.append((0, col, True))
                    elif pipe == 'VC':
                        actions.append((0, col, False))
                    elif pipe == 'VE':
                        actions.append((0, col, False))
                    elif pipe == 'VD':
                        actions.append((0, col, True))
                    else:
                        actions.append((0, col, True))
                        actions.append((0, col, False))

                # fronteira de baixo
                elif row == last_idx:
                    pipe = state.board.get_value(last_idx, col)
                    if pipe == 'BC' or pipe == 'LH':
                        continue
                    elif pipe == 'BE':
                        actions.append((last_idx, 0, True))
                    elif pipe == 'BD':
                        actions.append((last_idx, 0, False))
                    elif pipe == 'FE':
                        actions.append((last_idx, 0, True))
                    elif pipe == 'FD':
                        actions.append((last_idx, 0, False))
                    elif pipe == 'VB':
                        actions.append((0, col, False))
                    elif pipe == 'VC':
                        actions.append((0, col, True))
                    elif pipe == 'VE':
                        actions.append((0, col, True))
                    elif pipe == 'VD':
                        actions.append((0, col, False))
                    else:
                        actions.append((0, col, True))
                        actions.append((0, col, False))

                # peças interiores
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

