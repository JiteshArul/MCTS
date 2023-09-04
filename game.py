'''Implementation of TicTacToe for MCTS algorithms'''
import numpy as np

class TicTacToe:
    '''
    Class tictactoe has multiple methods to retrive values such as states and termination status
    '''
    def __init__(self) -> None:
        self.row_len = 3
        self.column_len = 3
        self.state_size = self.row_len * self.column_len

    def initial_state(self):
        '''
        returns [[0,0,0],[0,0,0],[0,0,0]] for 3X3 matrix
        '''
        return np.zeros((self.row_len,self.column_len))
    def next_state(self,state,action,player):
        '''Returns the next state in tictactoe'''
        row = action//self.column_len
        column = action%self.column_len
        state[row,column] = player
        return state
    def valid_moves(self,state):
        '''Returns a list of valid moves'''
        return (state.reshape(-1) == 0).astype(np.uint8)
    def check_win(self, state, action):
        '''Checks if any of the tictactoe win condition is satisfied'''
        row = action // self.column_len
        column = action % self.column_len
        player = state[row, column]

        return (
            np.sum(state[row, :]) == player * self.column_len
            or np.sum(state[:, column]) == player * self.row_len
            or np.sum(np.diag(state)) == player * self.row_len
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_len
        )
    def get_value_and_terminated(self, state, action):
        '''Checks if the game is terminated and returns the value'''
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.valid_moves(state)) == 0:
            return 0, True
        return 0, False
    def get_opponent(self, player):
        '''Returns opponent since player 1 has value 1, player 2 has value -1'''

        return -player
