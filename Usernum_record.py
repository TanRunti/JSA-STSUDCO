
# Record the number of users
class user_name_set:
    usernum = 0

    def __init__(self):
        self.a = 0

    def receive_user_num(self, user_num):
        user_name_set.usernum = user_num

    def get_user_num(self):
        return int(user_name_set.usernum)
