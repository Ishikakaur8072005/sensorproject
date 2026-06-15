import sys  # module which provides access to interpreter-related functions and variables

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()

    return (
        f"Error occurred in python script "
        f"[{exc_tb.tb_frame.f_code.co_filename}] "
        f"at line [{exc_tb.tb_lineno}] "
        f"error message [{error}]"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message,
            error_detail=error_detail
        )

    def __str__(self):
        return self.error_message