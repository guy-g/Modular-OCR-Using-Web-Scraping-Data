

class ProjectIsAlreadyExists(Exception):
    def __init__(self):
        super().__init__('Project is already exists!')


class ProjectIsNotExists(Exception):
    def __init__(self):
        super().__init__('Project is not exists!')

