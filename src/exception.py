import sys
import logging
import logger

class CustomeException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail)
    
    def error_message_detail(self, error, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occurred in python script name [{0}] line [{1}] error: [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error)
        )
        return error_message

    def __str__(self):
        return self.error_message

if __name__ == '__main__':
   pass 