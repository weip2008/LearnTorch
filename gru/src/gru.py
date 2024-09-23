import logging

class Logger:
    """
    Available levels:
        logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    """
    def __init__(self, log_file='app.log', log_level=logging.INFO):
        # Create a custom logger
        self.logger = logging.getLogger(__name__)
        
        # Set the default log level
        self.logger.setLevel(log_level)
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatters and add it to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()  # This will output to the console (screen)
        console_handler.setLevel(logging.INFO)

        # Add handlers to the logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
    
    def setLevel(self, log_level):
        self.logger.setLevel(log_level)

    def get_level(self):
        """
        Returns the current log level in human-readable form.
        """
        level = self.logger.getEffectiveLevel()
        # return logging.getLevelName(level)  # Returns the name of the log level (e.g., 'INFO')
        return level  # Returns the name of the log level (e.g., 'INFO')

# Example usage:
if __name__ == '__main__':
    log = Logger(log_file='my_log_file.log') # default log level: INFO
    log.info("This is an info message")
    log.error("This is an error message")
    log.setLevel(logging.WARNING)
    log.info("This is an info message")
    log.error("This is an error message")

