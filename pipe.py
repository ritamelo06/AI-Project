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

directions = {
    "FC": (1, 0, 0, 0), "FD": (0, 1, 0, 0), "FB": (0, 0, 1, 0), "FE": (0, 0, 0, 1),
    "BC": (1, 1, 0, 1), "BD": (1, 1, 1, 0), "BB": (0, 1, 1, 1), "BE": (1, 0, 1, 1),
    "VC": (1, 0, 0, 1), "VD": (1, 1, 0, 0), "VB": (0, 1, 1, 0), "VE": (0, 0, 1, 1),
    "LH": (0, 1, 0, 1), "LV": (1, 0, 1 ,0)
}

class PipeManiaState:
    state_id = 0
    next_position = 0 
    goal_tested = False

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id


class Board:
    """Representação interna de um tabuleiro de PipeMania."""
    size_n = 0
    number_correct = 0

    def __init__(self, data):
        self.data = data
        self.correct_position  = np.zeros((self.size_n, self.size_n))

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
            return None, self.get_value(row, col_right)
        elif col_right > self.size_n - 1:
            return self.get_value(row, col_left), None
        else:
            return self.get_value(row, col_left), self.get_value(row, col_right)

    def print_board(self) -> str:
        for row in range(self.size_n):
            print("\t".join(self.data[row]))

    def copy_board(self):
        new_board = []
        for row in range(self.size_n):
            new_line = []
            for col in range(self.size_n):
                new_line.append(self.data[row][col])
            new_board.append(new_line)
        np_array = np.array(new_board)
        new_board_instance = Board(np_array)
        
        # Criar uma copia da matriz de flags
        new_correct_position = np.zeros((self.size_n, self.size_n))
        for i in range(self.size_n):
            for j in range(self.size_n):
                new_correct_position[i, j] = self.correct_position[i, j]

        new_board_instance.correct_position = new_correct_position  # Atribuir a cópia da matriz de flags ao novo objeto Board
        new_board_instance.number_correct = self.number_correct
   
        return new_board_instance

    def fixate_position(self, row, col):
        if not self.correct_position[row][col]:
            self.correct_position[row][col] = 1
            self.number_correct += 1

    def deduce_pipe(self, row, col):
        top_adj, bottom_adj = self.adjacent_vertical_values(row, col)
        left_adj, right_adj = self.adjacent_horizontal_values(row, col)
        current_pipe = self.get_value(row, col)
        options_top, options_bottom, options_left, options_right, list_aux = [], [], [], [], []
        
        # top adjacent 
        if top_adj == None or (self.correct_position[row - 1][col] and not directions[top_adj][2]): # nao tem vizinho de cima ou o vizinho de cima já está na posição certa e não tem direção para baixo
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]) and not direction[0]:
                    options_top.append(pipe)
                    #a peça NAO pode ter saida para cima
            list_aux.append(options_top)
        elif top_adj != None and self.correct_position[row - 1][col] and directions[top_adj][2]:  # a peça de cima já está na posição certa e tem direcao para baixo
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]) and direction[0]:
                    options_top.append(pipe)
                    # tem que ter para cima
            list_aux.append(options_top)

        # bottom adjacent
        if bottom_adj == None or (self.correct_position[row + 1][col] and not directions[bottom_adj][0]): # nao tem vizinho de baixo ou o vizinho de baixo já está na posição certa e não tem direção para cima
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]) and not direction[2]:
                    options_bottom.append(pipe)
                    #a peça NAO pode ter saida para baixo 
            list_aux.append(options_bottom)
        elif bottom_adj != None and self.correct_position[row + 1][col] and directions[bottom_adj][0]: # a peça de baixo já está na posição certa e tem direcao p cima
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]) and direction[2]:
                    options_bottom.append(pipe)
                    # tem que ter para baixo
            list_aux.append(options_bottom)


        # left adjacent
        if left_adj == None or (self.correct_position[row][col - 1] and not directions[left_adj][1]): # nao tem vizinho da esquerda ou o vizinho da esquerda já está na posição certa e não tem direção para a direita
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]) and not direction[3]:
                    options_left.append(pipe)
                    #a peça NAO pode ter saida para a esquerda
            list_aux.append(options_left)
        elif left_adj != None and self.correct_position[row][col - 1] and directions[left_adj][1]: # a peça da esquerda já está na posição certa e tem direcao para a direita
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]) and direction[3]:
                    options_left.append(pipe)
                # tem que ter para esquerda
            list_aux.append(options_left)

           
        # right adjacent 
        if right_adj == None or (self.correct_position[row][col + 1] and not directions[right_adj][3]): # nao tem vizinho da direita ou o vizinho da direita já está na posição certa e não tem direção para a esquerda
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]) and not direction[1]:
                    options_right.append(pipe)
                #a peça NAO pode ter saida para a direita
            list_aux.append(options_right)
        elif right_adj != None and self.correct_position[row][col + 1] and directions[right_adj][3]: # a peça da direita já está na posição certa e tem direcao para a esquerda
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]) and direction[1]:
                    options_right.append(pipe)
                    # tem que ter para direita
            list_aux.append(options_right)

        # se houver deducoes possiveis
        if len(list_aux) != 0:
            final_options = set(list_aux[0])
            for option_list in list_aux[1:]:
                final_options = final_options.intersection(set(option_list)) # fazer a interseção das listas de opções
            final_options_list = list(final_options)

            if len(final_options_list) == 1:     # se só houver uma opção, então temos a certeza da posição
                self.fixate_position(row, col)   # fixar o pipe
            return final_options_list
        
        # se não há deduções, append todas as posições
        else:       
            final_options = []
            for pipe, direction in directions.items():
                if pipe.startswith(current_pipe[0]):
                    final_options.append(pipe)
            return final_options  

   
    def inferences(self):
        changes = True
        while changes:
            changes = False
            for row in range(self.size_n):
                for col in range(self.size_n):
                    if not self.correct_position[row][col]:
                        deductions = self.deduce_pipe(row, col)
                        if len(deductions) == 1:
                            changes = True
                            self.data[row][col] = deductions[0]

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
        np_array = np.array(processed_data)    
        board_instance = Board(np_array)
        board_instance.inferences()
        return board_instance

  
class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = PipeManiaState(board)
        super().__init__(self.initial)

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""   
        actions = []
    
        # verificar se já todas as coordenadas foram visitadas, matar a branch
        if state.next_position >= state.board.size_n ** 2:    
            return actions     
        
        # se todas as peças já estão marcadas com "1" e o goal_test deu False então queremos matar a branch
        elif (state.board.number_correct == state.board.size_n ** 2 and state.goal_tested):
            return actions
        
        else:
            row = state.next_position // state.board.size_n
            col = state.next_position % state.board.size_n
            state.next_position += 1

            # verificar se a peça já está na posição certa
            if state.board.correct_position[row][col]:
                actions.append((row, col, state.board.get_value(row, col)))  # movimento fantasma
                return actions
            
            # se nao estiver na posicao certa, calcular as deducoes
            else:
                #state.board.inferences()
                deductions = state.board.deduce_pipe(row, col) 
                if not len(deductions):  # se nao houver deducoes possiveis mata a branch
                    return actions
                else:
                    for i in range(len(deductions)):
                        actions.append((row, col, deductions[i]))
                    return actions
            
  
    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        row, col, pipe = action[0], action[1], action[2]
        new_board = state.board.copy_board()
        new_board.data[row][col] = pipe
        new_board.fixate_position(row, col)
        new_board.inferences()
        new_state = PipeManiaState(new_board)
        new_state.next_position = state.next_position         
        return new_state


    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema.""" 
        stack = [(0 , 0)]              
        visited_matrix = np.zeros((state.board.size_n, state.board.size_n))
        visited_count = 0
        state.goal_tested = True
    
        while stack:
            pipe_coord = stack.pop()                   
            row = pipe_coord[0]
            col = pipe_coord[1]     
            
            if not visited_matrix[row][col]:
                visited_matrix[row][col] = 1
                visited_count += 1
                pipe_dir = directions[state.board.get_value(row, col)]     #pipe_dir = (x,x,x,x)   
                
                # top
                if pipe_dir[0]:
                    adj_top = state.board.adjacent_vertical_values(row, col)[0]     # vizinho de cima
                    if adj_top == None:
                        return False
                    if directions[adj_top][2]:   # ver se o vizinho de cima tem pipe na direcao para baixo
                        stack.append((row - 1, col))
                    else:
                        return False
               
                # right
                if pipe_dir[1]:
                    adj_right = state.board.adjacent_horizontal_values(row, col)[1] # vizinho da direita
                    if adj_right == None:
                        return False
                    if directions[adj_right][3]:   # ver se o vizinho da direita tem pipe na direcao da esquerda
                        stack.append((row, col + 1))
                    else:
                        return False
                    
                # bottom
                if pipe_dir[2]:
                    adj_bottom = state.board.adjacent_vertical_values(row, col)[1] # vizinho de baixo
                    if adj_bottom == None:
                        return False
                    if directions[adj_bottom][0]:  # ver se o vizinho de baixo tem pipe na direcao de cima
                        stack.append((row + 1, col))
                    else:
                        return False

                # left
                if pipe_dir[3]:
                    adj_left = state.board.adjacent_horizontal_values(row, col)[0] # vizinho da esquerda
                    if adj_left == None:
                        return False
                    if directions[adj_left][1]:    # ver se o vizinho da esquerda tem pipe na direcao da direita
                        stack.append((row, col - 1))
                    else:
                        return False
       
        return visited_count == state.board.size_n ** 2

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

if __name__ == "__main__":
    board: Board = Board.parse_instance()
    problem: Problem = PipeMania(board)
    goal_node: Node = depth_first_tree_search(problem)
    goal_node.state.board.print_board()

    

    
    
  

  
    


