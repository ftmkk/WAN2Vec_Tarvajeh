from datetime import datetime, timedelta


class ResponseInPack:
    def __init__(self, line):
        format = "%Y-%m-%d %H:%M:%S"
        splitted = line[1:-1].replace('\\N', '""').split('","')
        self.pack_id = splitted[0]
        self.gender = splitted[1]
        self.age = splitted[2]
        self.start_time = splitted[3]
        self.local_start_time = datetime.strptime(self.start_time, format) + timedelta(0, 12600)
        self.finish_time = splitted[4]
        self.local_finish_time = None
        if self.finish_time != '':
            self.local_finish_time = datetime.strptime(self.finish_time, format) + timedelta(0, 12600)
        self.stimulus = splitted[5]
        self.response = splitted[6]
        self.duration = splitted[7]
        self.number = splitted[8]
        self.creation_time = splitted[9]
        self.local_creation_time = datetime.strptime(self.creation_time, format) + timedelta(0, 12600)
