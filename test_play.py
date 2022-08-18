class CheckersBoard:
    def __init__(self):
        self.board = [[0 for x in range(8)] for y in range(8)]
        self.board[3][3] = 1
        self.board[3][4] = 2
        self.board[4][3] = 2
        self.board[4][4] = 1
        self.turn = 1
        self.winner = 0
        self.king = 0
        self.king_count = 0
        self.king_count_2 = 0
        self.king_count_3 = 0
        self.king_count_4 = 0
        self.king_count_5 = 0
        self.king_count_6 = 0
        self.king_count_7 = 0
        self.king_count_8 = 0
        self.king_count_9 = 0
        self.king_count_10 = 0
        self.king_count_11 = 0
        self.king_count_12 = 0
        self.king_count_13 = 0
        self.king_count_14 = 0
        self.king_count_15 = 0
        self.king_count_16 = 0
        self.king_count_17 = 0
        self.king_count_18 = 0
        self.king_count_19 = 0
        self.king_count_20 = 0
        self.king_count_21 = 0
        self.king_count_22 = 0
        self.king_count_23 = 0
        self.king_count_24 = 0
        self.king_count_25 = 0
        self.king_count_26 = 0
        self.king_count_27 = 0
        self.king_count_28 = 0
        self.king_count_29 = 0
        self.king_count_30 = 0
        self.king_count_31 = 0
        self.king_count_32 = 0
        self.king_count_33 = 0
        self.king_count_34 = 0
        self.king_count_35 = 0
        self.king_count_36 = 0

    def get_board(self):
        return self.board

    def get_turn(self):
        return self.turn

    def get_winner(self):
        return self.winner

    def get_king(self):
        return self.king

    def get_king_count(self):
        return self.king_count

    def get_king_count_2(self):
        return self.king_count_2

    def get_king_count_3(self):
        return self.king_count_3

    def get_king_count_4(self):
        return self.king_count_4
