import torch
import pygame
import chess
import threading
import math
import random
import numpy as np


class Node:
    # initialize with state, action, parent, and exploration weight
    def __init__(self, state, action=None, parent=None, c=1):
        # store state, action, and parent
        self.state = state
        self.action = action
        self.parent = parent
        # initialize children list
        self.children = []
        # initialize visit count and value estimate
        self.visit_count = 0
        self.value = 0
        # store exploration weight
        self.c = c

    # calculate UCB value
    def ucb(self):
        # if node is unvisited, return infinity
        if self.visit_count == 0:
            return float("inf")
        # else, return UCB formula
        else:
            return self.value / self.visit_count + self.c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
    pass

class MCTS:
    # initialize with model and input tensor
    def __init__(self, model, input_tensor):
        # create root node with input tensor as state
        self.root = Node(input_tensor)
        # store model
        self.model = model

    # select method
    def select(self):
        # start from root node
        node = self.root
        # loop until leaf node is reached
        while len(node.children) > 0:
            # get UCB values for all children
            ucb_values = [child.ucb() for child in node.children]
            # get index of child with maximum UCB value
            max_index = ucb_values.index(max(ucb_values))
            # select child with maximum UCB value
            node = node.children[max_index]
        # return leaf node
        return node

    # expand method
    def expand(self, node):
        # get board state from node
        board_state = node.state
        # get legal moves from board state
        legal_moves = board_state.legal_moves()
        # loop over legal moves
        for move in legal_moves:
            # apply move to board state and get new board state
            new_board_state = board_state.apply_move(move)
            # encode new board state as a tensor
            new_input_tensor = encode_board(new_board_state)
            # create child node with new board state and move as action
            child_node = Node(new_input_tensor, move)
            # add child node to parent node's children list
            node.children.append(child_node)

    # simulate method
    def simulate(self, node):
        # get board state from node
        board_state = node.state
        # loop until game is over
        while board_state.game_result is None:
            # get legal moves from board state
            legal_moves = board_state.legal_moves()
            # choose a random move from legal moves
            random_move = random.choice(legal_moves)
            # apply move to board state and get new board state
            board_state = board_state.apply_move(random_move)
        # return game result as reward (-1, 0, or 1)
        return board_state.game_result

    # update method
    def update(self, node, reward):
        # loop until root node is reached
        while node is not None:
            # increment visit count by 1
            node.visit_count += 1
            # update value estimate by adding reward divided by visit count
            node.value += (reward - node.value) / node.visit_count
            # go to parent node
            node = node.parent

    # run MCTS simulation for some iterations
    def run(self, iterations):
        # loop for given number of iterations
        for i in range(iterations):
            # select a leaf node
            selected_node = self.select()
            # expand the selected node if it is not terminal
            if selected_node.state.game_result is None:
                self.expand(selected_node)
                # choose a random child to simulate from if there are any children (otherwise stay at selected node)
                if len(selected_node.children) > 0:
                    selected_node = random.choice(selected_node.children)
            # simulate a random playout from the selected node and get reward
            reward = self.simulate(selected_node)
            # update all nodes along the path with reward
            self.update(selected_node, reward)

    # return a move probability vector based on visit counts of root's children 
    def get_move_probs(self):
        # get the visit counts of all children
        visit_counts = [child.visit_count for child in self.root.children]
        # convert to numpy array
        visit_counts = np.array(visit_counts)
        # normalize by dividing by the sum
        visit_counts = visit_counts / np.sum(visit_counts)
        # return the normalized vector
        return visit_counts
    pass

    def train_model(self):
        # create an empty list to store the game data
        game_data = []
        # loop for a given number of games
        for i in range(num_games):
            # create a new board object for each game
            board = chess.Board()
            threading.Thread(target=boardRender, args=(board,)).start()
            # create a new MCTS object for each game
            mcts = MCTS(model, encode_board(board))
            # create an empty list to store the board states and move probabilities for each game
            states = []
            probs = []
            # loop until the game is over
            while not board.is_game_over():
                # run the MCTS simulation for some iterations
                mcts.run(iterations)
                # get the move probability vector from the MCTS object
                move_probs = mcts.get_move_probs()
                # append the current board state and move probability vector to the lists
                states.append(encode_board(board))
                probs.append(move_probs)
                # apply the move with the highest probability to the board state
                mcts.apply_best_move()
            # get the game result as a reward (-1, 0, or 1)
            reward = board.game_result()
            # invert the reward for every other state in the list (because of alternating turns)
            rewards = [reward if i % 2 == 0 else -reward for i in range(len(states))]
            # append the states, probs, and rewards lists as a tuple to the game data list
            game_data.append((states, probs, rewards))
        # convert the game data list to numpy arrays
        states = np.concatenate([np.array(x[0]) for x in game_data])
        probs = np.concatenate([np.array(x[1]) for x in game_data])
        rewards = np.concatenate([np.array(x[2]) for x in game_data])
        # train the network using the states, probs, and rewards arrays as inputs and targets
        model.fit(states, [probs, rewards], epochs=epochs, batch_size=batch_size)
        # save the model using torch.save method
        torch.save(model, "model.pt")

class ChessNet():
    # initialize with input size and output size
    def __init__(self, input_size, output_size):
        # call the parent constructor
        super(ChessNet, self).__init__()
        # define the hidden layer size
        hidden_size = 768
        # define the first linear layer with input size and hidden size
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        # define the second linear layer with hidden size and output size
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        # define the softmax layer for the policy output
        self.softmax = torch.nn.Softmax(dim=1)
        # define the tanh layer for the value output
        self.tanh = torch.nn.Tanh()

    # define the forward pass
    def forward(self, x):
        # pass the input through the first linear layer and apply relu activation
        x = torch.nn.relu(self.linear1(x))
        # pass the output through the second linear layer
        x = self.linear2(x)
        # split the output into two parts: one for policy and one for value
        policy, value = x[:, :-1], x[:, -1]
        # apply softmax to the policy part
        policy = self.softmax(policy)
        # apply tanh to the value part
        value = self.tanh(value)
        # return policy and value as a tuple
        return policy, value

def boardRender(board):
    pygame.init()
    size = width, height = 640, 640
    screen = pygame.display.set_mode(size)

    while True:
        for square in chess.SQUARES:
            x, y = chess.square_file(square), 7 - chess.square_rank(square)
            color = (255, 206, 158) if (x + y) % 2 else (209, 139, 71)
            rect = pygame.Rect(x * width // 8, y * height // 8, width // 8, height // 8)
            pygame.draw.rect(screen, color, rect)

        for piece in board.piece_map().items():
            x, y = chess.square_file(piece[0]), 7 - chess.square_rank(piece[0])
            img = pygame.image.load(f"BigProjects/chessAI/pieces/{piece[1].symbol()}.png")
            img = pygame.transform.scale(img, (width // 8 - 10, height // 8 - 10))
            screen.blit(img, (x * width // 8 + 5, y * height // 8 + 5))

        pygame.display.flip()
    pass

def move(board, move_string):
    move = chess.Move.from_uci(move_string)
    if move in board.legal_moves:
        board.push(move)
    pass

def encode_board(board):
    # encode board state as a 8x8x12 binary tensor
    # see https://github.com/geochri/AlphaZero_Chess/blob/master/encoder_decoder.py for details
    # create an empty tensor of shape (8, 8, 12)
    input_tensor = torch.zeros((8, 8, 12))
    # loop over all squares on the board
    for square in chess.SQUARES:
        # get the piece on the square
        piece = board.piece_at(square)
        # if there is a piece
        if piece:
            # get the color and type of the piece
            color = piece.color
            piece_type = piece.piece_type
            # get the rank and file of the square
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            # set the corresponding channel of the tensor to 1
            input_tensor[rank][file][piece_type - 1 + (6 if color else 0)] = 1
    # return the input tensor
    return input_tensor
    pass

def decode_move(move):
    # decode move vector as a chess move object
    # see https://github.com/geochri/AlphaZero_Chess/blob/master/encoder_decoder.py for details
    # get the from and to squares from the move vector
    from_square = move[0] * 8 + move[1]
    to_square = move[2] * 8 + move[3]
    # create a chess move object with the from and to squares
    chess_move = chess.Move(from_square, to_square)
    # return the chess move object
    return chess_move
    pass

def evaluate(board):
    # encode board state as a tensor
    input_tensor = encode_board(board)
    # pass input tensor to network and get output tensor
    output_tensor = model(input_tensor)
    # decode output tensor as a scalar score
    score = decode_score(output_tensor)
    # return score
    return score
    pass

def decode_score(score):
    # decode score tensor as a scalar value
    # squeeze the tensor to remove any extra dimensions
    score = torch.squeeze(score)
    # get the scalar value from the tensor using item method
    score = score.item()
    # return the scalar value
    return score


# define epochs, batch_size, iterations, and num_games as global variables
epochs = 10
batch_size = 1
iterations = 100
num_games = 1000

#create a model
model = ChessNet(768, 4097)

#i have no clue why, but this makes it work
#most likelly introduces an awful bug
mcts = MCTS(model, encode_board(chess.Board()))

#train model
mcts.train_model()
